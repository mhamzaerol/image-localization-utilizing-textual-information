from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd

from classification import utils_global
from classification.s2_utils import Partitioning, Hierarchy
from classification.dataset import StreetViewDataset

import torch.nn as nn


class OcrEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        linear1 = nn.Linear(input_size, hidden_size)
        linear2 = nn.Linear(hidden_size, out_size)
        self.model = nn.Sequential(linear1, nn.ReLU(), linear2, nn.ReLU())

    def forward(self, x):
        return self.model(x)


class CombinedClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        linear1 = nn.Linear(input_size, hidden_size)
        linear2 = nn.Linear(hidden_size, hidden_size)
        linear3 = nn.Linear(hidden_size, out_size)
        self.model = nn.Sequential(linear1, nn.ReLU(), linear2, nn.ReLU(), linear3)
    
    def forward(self, x):
        return self.model(x)


class MultiPartitioningClassifier(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

        self.partitionings, self.hierarchy = self.__init_partitionings()
        self.__build_model()

    def __init_partitionings(self):

        partitionings = []
        for shortname, path in zip(
            self.hparams.partitionings["shortnames"],
            self.hparams.partitionings["files"],
        ):
            partitionings.append(Partitioning(Path(path), shortname, skiprows=0))

        if len(self.hparams.partitionings["files"]) == 1:
            return partitionings, None

        return partitionings, Hierarchy(partitionings)

    def __build_model(self):
        logging.info("Build model")
        
        ocr_params = self.hparams.ocr_params

        model, model_out_size = utils_global.build_base_model(self.hparams.arch)
    
        ocr_enc_out_size = 0

        if ocr_params['use_ocr']:
            ocr_encoder = OcrEncoder(
                ocr_params['ocr_encoder_params']['in_size'], 
                ocr_params['ocr_encoder_params']['hidden_size'],
                ocr_params['ocr_encoder_params']['out_size']
            )
            ocr_enc_out_size = ocr_params['ocr_encoder_params']['out_size']
        

        classifier = torch.nn.ModuleList(
            [
                CombinedClassifier(
                    model_out_size + ocr_enc_out_size, 
                    self.hparams.classifier_hidden_size, 
                    len(self.partitionings[i])
                )
                for i in range(len(self.partitionings))
            ]
        )

        if self.hparams.weights:
            logging.info("Load weights from pre-trained model")
            model, _ = utils_global.load_weights_if_available(
                model, None, self.hparams.weights
            )
            if self.hparams.frozen:        
                logging.info("Freeze weights of the pre-trained model")
                for param in model.parameters():
                    param.requires_grad = False

        self.model = model
        if ocr_params['use_ocr']:
            self.ocr_encoder = ocr_encoder
        self.classifier = classifier

    def forward(self, x, ocr_features):
        fv = self.model(x) 
        if self.hparams.ocr_params['use_ocr']:
            ocr_features = self.ocr_encoder(ocr_features)
            cv = torch.cat([fv, ocr_features], dim=1)            
        else:
            cv = fv
        yhats = [self.classifier[i](cv) for i in range(len(self.partitionings))]
        return yhats

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        images, ocr_features, target = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images, ocr_features)

        # individual losses per partitioning
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # stats
        losses_stats = {
            f"loss_train/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }
        for metric_name, metric_value in losses_stats.items():
            self.log(metric_name, metric_value, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, **losses_stats}

    def validation_step(self, batch, batch_idx):
        images, ocr_features, target, true_lats, true_lngs = batch

        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images, ocr_features)

        # loss calculation
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        # log top-k accuracy for each partitioning
        individual_accuracy_dict = utils_global.accuracy(
            output, target, [p.shortname for p in self.partitionings]
        )
        # log loss for each partitioning
        individual_loss_dict = {
            f"loss_val/{p}": l
            for (p, l) in zip([p.shortname for p in self.partitionings], losses)
        }

        # log GCD error@km threshold
        distances_dict = {}

        if self.hierarchy is not None:
            hierarchy_logits = [
                yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(output)
            ]
            hierarchy_logits = torch.stack(hierarchy_logits, dim=-1,)
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        pnames = [p.shortname for p in self.partitionings]
        if self.hierarchy is not None:
            pnames.append("hierarchy")
        for i, pname in enumerate(pnames):
            # get predicted coordinates
            if i == len(self.partitionings):
                i = i - 1
                pred_class_indexes = torch.argmax(hierarchy_preds, dim=1)
            else:
                pred_class_indexes = torch.argmax(output[i], dim=1)
            pred_latlngs = [
                self.partitionings[i].get_lat_lng(idx)
                for idx in pred_class_indexes.tolist()
            ]
            pred_lats, pred_lngs = map(list, zip(*pred_latlngs))
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            # calculate error
            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats.type_as(pred_lats),
                true_lngs.type_as(pred_lats),
            )
            distances_dict[f"gcd_{pname}_val"] = distances

        output = {
            "loss_val/total": loss,
            **individual_accuracy_dict,
            **individual_loss_dict,
            **distances_dict,
        }
        return output

    def validation_epoch_end(self, outputs):
        pnames = [p.shortname for p in self.partitionings]

        # top-k accuracy and loss per partitioning
        loss_acc_dict = utils_global.summarize_loss_acc_stats(pnames, outputs)

        # GCD stats per partitioning
        gcd_dict = utils_global.summarize_gcd_stats(pnames, outputs, self.hierarchy)

        metrics = {
            "val_loss": loss_acc_dict["loss_val/total"],
            **loss_acc_dict,
            **gcd_dict,
        }
        for metric_name, metric_value in metrics.items():
            self.log(metric_name, metric_value, logger=True)

    def _main_inference(self, batch):
        images, ocr_features, target, true_lats, true_lngs = batch

        # forward pass
        yhats = self(images, ocr_features)

        hierarchy_preds = None
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        return yhats, true_lats, true_lngs, hierarchy_preds

    def inference(self, batch):

        yhats, true_lats, true_lngs, hierarchy_preds = self._main_inference(batch)

        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        pred_class_dict = {}
        pred_lat_dict = {}
        pred_lng_dict = {}
        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            pred_lat_dict[pname] = pred_lats
            pred_lng_dict[pname] = pred_lngs
            pred_class_dict[pname] = pred_classes

        return list(zip(true_lats, true_lngs)), pred_class_dict, pred_lat_dict, pred_lng_dict

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        yhats, true_lats, true_lngs, hierarchy_preds = self._main_inference(batch)

        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            true_lats = torch.tensor(true_lats, dtype=torch.float)
            true_lngs = torch.tensor(true_lngs, dtype=torch.float)

            distances = utils_global.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats,
                true_lngs,
            )
            distances_dict[pname] = distances

        return distances_dict

    def test_epoch_end(self, outputs):
        result = utils_global.summarize_test_gcd(
            [p.shortname for p in self.partitionings], outputs, self.hierarchy
        )
        return {**result}

    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_feature_extrator,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_feature_extrator, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def train_dataloader(self):

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = StreetViewDataset(
            data_path=self.hparams.train_dir,
            ocr_json_path=self.hparams.ocr_params['train_ocr_json_path'],
            ocr_feat_path=self.hparams.ocr_params['train_ocr_feat_path'],
            label_path=self.hparams.train_label_mapping,
            use_ocr=self.hparams.ocr_params['use_ocr'],
            shuffle=True,
            transformation=tfm,
            give_latlng=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):

        tfm = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = StreetViewDataset(
            data_path=self.hparams.val_dir,
            ocr_json_path=self.hparams.ocr_params['val_ocr_json_path'],
            ocr_feat_path=self.hparams.ocr_params['val_ocr_feat_path'],
            label_path=self.hparams.val_label_mapping,
            use_ocr=self.hparams.ocr_params['use_ocr'],
            shuffle=False,
            transformation=tfm,
            give_latlng=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
        )

        return dataloader


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("config/baseM.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    utils_global.check_is_valid_torchvision_architecture(model_params["arch"])

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init classifier
    model = MultiPartitioningClassifier(hparams=Namespace(**model_params))
    logging.info("Model initialized!")

    logger = pl.loggers.TensorBoardLogger(save_dir=str(out_dir), name="tb_logs")
    logging.info("Logger initialized!")
    checkpoint_dir = out_dir / "ckpts" / "{epoch:03d}-{val_loss:.4f}"
    checkpointer = pl.callbacks.model_checkpoint.ModelCheckpoint(checkpoint_dir)
    logging.info("Checkpointer initialized!")

    progress_bar_refresh_rate = 0
    if args.progbar:
        progress_bar_refresh_rate = 1

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        val_check_interval=model_params["val_check_interval"],
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        log_every_n_steps=1,
    )
    logging.info("Trainer initialized!")

    trainer.fit(model)


if __name__ == "__main__":
    main()
