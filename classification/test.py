from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import torch
import pytorch_lightning as pl
import pandas as pd


from classification.train_base import MultiPartitioningClassifier
import torchvision
from classification.dataset import StreetViewDataset


def parse_args():
    args = ArgumentParser()
    # model
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/base_M/epoch=014-val_loss=18.4833.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams",
        type=Path,
        default=Path("models/base_M/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        required=True,
        help="Whitespace separated list of image folders to evaluate",
    )
    args.add_argument(
        "--ocr_json_dir",
    )
    args.add_argument(
        "--ocr_feat_dir",
    )
    args.add_argument(
        "--label_dir",
        required=True,
    )
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32), half precision (16)",
    )
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


args = parse_args()
print("Load model from checkpoint", args.checkpoint)
model = MultiPartitioningClassifier.load_from_checkpoint(
    checkpoint_path=str(args.checkpoint),
    hparams_file=str(args.hparams),
    map_location=None,
)

hparams = model.hparams

if args.gpu and torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = None
trainer = pl.Trainer(gpus=args.gpu, precision=args.precision)

print("Init Testset")

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
    data_path=args.image_dir,
    ocr_json_path=args.ocr_json_dir,
    ocr_feat_path=args.ocr_feat_dir,
    label_path=args.label_dir,
    use_ocr=hparams.ocr_params['use_ocr'],
    shuffle=False,
    transformation=tfm,
    give_latlng=True,
)
dataloader = [torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)]

print("Testing")
r = trainer.test(model, test_dataloaders=dataloader, verbose=False)
# formatting results
dfs = []
for results, name in zip(r, [Path(args.image_dir).stem]):
    df = pd.DataFrame(results).T
    df["dataset"] = name
    df["partitioning"] = df.index
    df["partitioning"] = df["partitioning"].apply(lambda x: x.split("/")[-1])
    df.set_index(keys=["dataset", "partitioning"], inplace=True)
    dfs.append(df)

df = pd.concat(dfs)

print(df)
fout = Path(args.checkpoint).parent / ("test-" + "_".join(
    [str(Path(args.image_dir).stem)]) + ".csv")
print("Write to", fout)
df.to_csv(fout)
