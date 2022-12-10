import os
import sys
import re
from math import ceil
from typing import Dict, List, Tuple, Union
from io import BytesIO
import random
from pathlib import Path
from multiprocessing import Pool
import json

import pandas as pd
from PIL import Image
import torchvision
import torch
from torch.utils.data import Dataset
import msgpack
import numpy as np

class StreetViewDataset(Dataset):
    def __init__(
        self,
        data_path = "/root/dataset-streetview/",
        ocr_json_path = "/root/easy-ocr/feat_id.json",
        ocr_feat_path = "/root/easy-ocr/ocr_feats_final.npy",
        label_path = "/root/reproduce/GeoEstimation/resources/streetview_labels.json",
        use_ocr = True,
        transformation = None,
        give_latlng = False,
    ):
        super(StreetViewDataset, self).__init__()
        root = Path(data_path)
        self.use_ocr = use_ocr
        self.img_paths = list(root.glob("*.png"))
        self.img_names = [x.name for x in self.img_paths]
        self.latlngs = [np.array([float(x.stem.split(",")[0]), float(x.stem.split(",")[1])]) for x in self.img_paths]
        self.transformation = transformation
        with open(ocr_json_path, "r") as ocr_json:
            self.ocr_index_map = json.load(ocr_json)
        with open(label_path, "r") as label_json:
            self.label_map = json.load(label_json)
        self.ocr_features = np.load(ocr_feat_path)
        self.give_latlng = give_latlng

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")

        # this was the line in the original version
        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        if self.transformation is not None:
            img = self.transformation(img)

        ocr_features = None

        if self.use_ocr:
            ocr_feat_index = self.ocr_index_map[str(self.img_names[idx])]
            ocr_features = self.ocr_features[ocr_feat_index]

        if give_latlng:
            return img, ocr_features, self.labels[idx], *self.latlngs[idx]

        return img, ocr_features, self.labels[idx]


if __name__ == "__main__":
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
    dataset = StreetViewDataset(transformation = tfm)
    from torch.utils.data import DataLoader
    dl = DataLoader(dataset, batch_size=32, num_workers=0)

    a = next(iter(dl))
    for el in a:
        print(el.shape)