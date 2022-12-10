import easyocr
import cv2
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
import torchvision
import json
from pathlib import Path
import argparse
import yaml

from train_dataset import StreetViewDataset

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to the config file")
    
    args = parser.parse_args()
    return args

def download_model(lang):
    print(lang)
    reader = easyocr.Reader([lang])

def res_to_feat(result):
    feat = []
    EPS=1e-6
    for res in result:
        maxi = EPS
        for text in res:
            if any(char.isdigit() for char in text[1]) or len(text[1]) < 3:
                continue
            maxi = max(maxi, text[2])
        feat.append(maxi)

    feat_np = np.array(feat)
    return feat_np

def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    lang_list = []
    with open("easy-ocr/lang_list.txt", "r") as f:
        for line in f:
            lang_list.append(line.strip())
    print(f"Total of {len(lang_list)} languages")

    config = config["model_params"]

    for dataset_type in ["test", "train"]:

        if dataset_type == "test":
            dataset_var = "val"
        else:
            dataset_var = dataset_type

        batch_size = 32
        img_paths = list(Path(config[f"{dataset_var}_dir"]).glob("*.png"))
        img_paths = [str(x) for x in img_paths]

        id_json = dict()
        for i in range(len(img_paths)):
            key = img_paths[i].split("/")[-1]
            id_json[key] = i
        with open(config["ocr_params"][f"{dataset_var}_ocr_json_path"], "w") as f:
            json.dump(id_json, f, indent=4)

        idxs = []
        all_feats = []
        for lang in lang_list:
            print(lang)
            feats = []
            model = easyocr.Reader([lang], gpu=True)
            for i in tqdm(range(0, len(img_paths), batch_size)):
                paths_batch = img_paths[i:i+batch_size]
                result = model.readtext_batched(paths_batch, n_width=768, n_height=768)
                feat = res_to_feat(result)
                feats.append(feat)

            feats = np.expand_dims(np.concatenate(feats), 1)
            all_feats.append(feats)
            np.save(config["ocr_params"][f"{dataset_var}_ocr_feat_path"], np.concatenate(all_feats, 1))
        
        all_feats = np.concatenate(all_feats, 1)
        print(all_feats.shape)
        np.save(config["ocr_params"][f"{dataset_var}_ocr_feat_path"], all_feats)


if __name__ == "__main__":
    main()