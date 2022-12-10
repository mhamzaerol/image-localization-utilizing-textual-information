import csv
import os
import logging
import sys
import argparse
from time import time
from functools import partial
from multiprocessing import Pool
from collections import Counter

import pandas as pd
import s2sphere as s2

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config"
    )
    parser.add_argument(
        "-ci",
        "--column_img_path",
        type=str,
        default="IMG_ID",
        help="column name image id / path",
    )
    parser.add_argument(
        "-clat", "--column_lat", type=str, default="LAT", help="column name latitude"
    )
    parser.add_argument(
        "-clng", "--column_lng", type=str, default="LON", help="column name longitude"
    )

    parser.add_argument(
        "--lvl_min",
        type=int,
        required=False,
        default=2,
        help="Minimum partitioning level",
    )
    parser.add_argument(
        "--lvl_max",
        type=int,
        required=False,
        default=30,
        help="Maximum partitioning level",
    )

    args = parser.parse_args()
    return args


def _init_parallel(img, level):
    cell = create_s2_cell(img[1], img[2])
    hexid = create_cell_at_level(cell, level)
    return [*img, hexid, cell]


def init_cells(img_container_0, level):

    start = time()
    f = partial(_init_parallel, level=level)
    img_container = []
    with Pool(8) as p:
        for x in p.imap_unordered(f, img_container_0, chunksize=1000):
            img_container.append(x)
    logging.debug(f"Time multiprocessing: {time() - start:.2f}s")
    start = time()
    h = dict(Counter(list(list(zip(*img_container))[3])))
    logging.debug(f"Time creating h: {time() - start:.2f}s")

    return img_container, h


def delete_cells(img_container, h, t_min):
    del_cells = {k for k, v in h.items() if v <= t_min}
    h = {k: v for k, v in h.items() if v > t_min}
    img_container_f = []
    for img in img_container:
        hexid = img[3]
        if hexid not in del_cells:
            img_container_f.append(img)
    return img_container_f, h


def gen_subcells(img_container_0, h_0, level, t_max):
    img_container = []
    h = {}
    for img in img_container_0:
        hexid_0 = img[3]
        if h_0[hexid_0] > t_max:
            hexid = create_cell_at_level(img[4], level)
            img[3] = hexid
            try:
                h[hexid] = h[hexid] + 1
            except:
                h[hexid] = 1
        else:
            try:
                h[hexid_0] = h[hexid_0] + 1
            except:
                h[hexid_0] = 1
        img_container.append(img)
    return img_container, h


def create_s2_cell(lat, lng):
    p1 = s2.LatLng.from_degrees(lat, lng)
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def create_cell_at_level(cell, level):
    cell_parent = cell.id().parent(level)
    hexid = cell_parent.to_token()
    return hexid


def write_output(img_container, h, out_dir):

    if not os.path.exists(out_p):
        os.makedirs(out_p)

    logging.info(f"Write to {out_dir}")
    with open(out_dir, "w") as f:
        cells_writer = csv.writer(f, delimiter=",")
        # write column names
        cells_writer.writerow(
            [
                "class_label",
                "hex_id",
                "imgs_per_cell",
                "latitude_mean",
                "longitude_mean",
            ]
        )

        # write dict
        i = 0
        cell2class = {}
        coords_sum = {}

        # generate class ids for each hex cell id
        for k in h.keys():
            cell2class[k] = i
            coords_sum[k] = [0, 0]
            i = i + 1

        # calculate mean GPS coordinate in each cell
        for img in img_container:
            coords_sum[img[3]][0] = coords_sum[img[3]][0] + img[1]
            coords_sum[img[3]][1] = coords_sum[img[3]][1] + img[2]

        # write partitioning information
        for k, v in h.items():
            cells_writer.writerow(
                [cell2class[k], k, v, coords_sum[k][0] / v, coords_sum[k][1] / v]
            )

def read_dataset(dataset_path):
    # get the names in the directory
    img_names = os.listdir(dataset_path)
    img_names = [x for x in img_names if x.endswith((".jpg", ".png"))]
    img_names_latlng = [(x, '.'.join(x.split('.')[:-1])) for x in img_names]
    img_names_latlng = [(x[0], *(x[1].split(','))) for x in img_names_latlng]
    img_names_latlng = [x for x in img_names_latlng if len(x) == 3] # remove it later
    img_names_latlng = [(x[0], float(x[1]), float(x[2])) for x in img_names_latlng]
    # make a data frame with columns: IMG_ID, LAT, LON
    df = pd.DataFrame(img_names_latlng, columns=["IMG_ID", "LAT", "LON"])
    return df


def main():
    # load arguments
    args = parse_args()
    level = logging.INFO

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=level,
    )

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = config["model_params"]

    output_files = config["partitionings"]["files"]
    output_dir = Path(output_files[0]).parent

    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(3):
        
        logging.info(f"Running {i}th partitioning ...")
        filename = output_files[i].split("/")[-1]
        _, img_min, img_max, __ = filename.split("_")
        img_min, img_max = int(img_min), int(img_max)

        # read dataset
        df = read_dataset(config["train_dir"])
        img_container = list(df.itertuples(index=False, name=None))
        num_images = len(img_container)
        logging.info("{} images available.".format(num_images))
        level = args.lvl_min

        # initialize
        logging.info("Initialize cells of level {} ...".format(level))
        start = time()
        img_container, h = init_cells(img_container, level)
        logging.info(f"Time: {time() - start:.2f}s - Number of classes: {len(h)}")

        logging.info("Remove cells with |img| < t_min ...")
        start = time()
        img_container, h = delete_cells(img_container, h, img_min)
        logging.info(f"Time: {time() - start:.2f}s - Number of classes: {len(h)}")

        logging.info("Create subcells ...")
        while any(v > img_max for v in h.values()) and level < args.lvl_max:
            level = level + 1
            logging.info("Level {}".format(level))
            start = time()
            img_container, h = gen_subcells(img_container, h, level, img_max)
            logging.info(f"Time: {time() - start:.2f}s - Number of classes: {len(h)}")

        logging.info("Remove cells with |img| < t_min ...")
        start = time()
        img_container, h = delete_cells(img_container, h, img_min)
        logging.info(f"Time: {time() - start:.2f}s - Number of classes: {len(h)}")
        logging.info(f"Number of images: {len(img_container)}")

        logging.info("Write output file ...")
        write_output(img_container, h, Path(output_files[i]))


if __name__ == "__main__":
    sys.exit(main())
