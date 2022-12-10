from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

from classification.train_base import MultiPartitioningClassifier
import torchvision
from classification.dataset import StreetViewDataset


def parse_args():
    args = ArgumentParser()
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
        type=Path,
        default=Path("resources/images/im2gps"),
        help="Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
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

print("Load model from ", args.checkpoint)

model = MultiPartitioningClassifier.load_from_checkpoint(
    checkpoint_path=str(args.checkpoint),
    hparams_file=str(args.hparams),
    map_location=None,
)

hparams = model.hparams

model.eval()
if args.gpu and torch.cuda.is_available():
    model.cuda()

print("Init dataloader")

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
    data_path=hparams.val_dir,
    ocr_json_path=hparams.ocr_params['val_ocr_json_path'],
    ocr_feat_path=hparams.ocr_params['val_ocr_feat_path'],
    label_path=hparams.val_label_mapping,
    use_ocr=hparams.ocr_params['use_ocr'],
    shuffle=False,
    transformation=tfm,
    give_latlng=True,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hparams.batch_size,
    num_workers=hparams.num_workers_per_loader,
    pin_memory=True,
)

print("Number of images: ", len(dataloader.dataset))
if len(dataloader.dataset) == 0:
    raise RuntimeError(f"No images found in {args.image_dir}")

rows = []
for X in tqdm(dataloader):
    if args.gpu:
        X[0] = X[0].cuda()
    true_coords, pred_classes, pred_latitudes, pred_longitudes = model.inference(X)
    for p_key in pred_classes.keys():
        for true_coord, pred_class, pred_lat, pred_lng in zip(
            true_coords,
            pred_classes[p_key].cpu().numpy(),
            pred_latitudes[p_key].cpu().numpy(),
            pred_longitudes[p_key].cpu().numpy(),
        ):
            rows.append(
                {
                    "coords": f'{true_coord[0]},{true_coord[1]}',
                    "p_key": p_key,
                    "pred_class": pred_class,
                    "pred_lat": pred_lat,
                    "pred_lng": pred_lng,
                }
            )
df = pd.DataFrame.from_records(rows)
df.set_index(keys=["coords", "p_key"], inplace=True)
print(df)
fout = Path(args.checkpoint).parent / f"inference_{args.image_dir.stem}.csv"
print("Write output to", fout)
df.to_csv(fout)
