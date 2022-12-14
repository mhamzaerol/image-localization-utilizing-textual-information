<div align="center">    
 
# GeoOCR: Predicting Geolocation of Images by Integrating Language Information

</div>

This Github page is built on top of the publication: ([Link](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)):


> Eric Müller-Budack, Kader Pustu-Iren, Ralph Ewerth:
"Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification".
In: *European Conference on Computer Vision (ECCV)*, Munich, Springer, 2018, 575-592.

The original repository of the publication could be accessed [here](https://github.com/TIBHannover/GeoEstimation).

## Basic Setup

After cloning the repository, stay in the base directory to run the commands. 
Run the following commands to setup the directory for any of the following procedures.
```
conda env create -f environment.yml
conda activate geoOCR
mkdir resources
mkdir resources/images
mkdir models
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/epoch.014-val_loss.18.4833.ckpt -P models/base_M
```

### Inference

To infer a pre-trained model, first download the relevant files and setup the folders:
```
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_streetview_labels.json -O resources/test_streetview_labels.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_ocr_feats_final.npy -O resources/test_ocr_feats_final.npy
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_feat_id.json -O resources/test_feat_id.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/s2_cells.zip -P resources/
unzip resources/s2_cells.zip -d resources/
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_streetview.zip -P resources/images
unzip resources/images/test_streetview.zip -d resources/images/test_streetview/
```

Assuming that we are inferring the model ``GeoOCR finetuned``. Then run the following commands:

```
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/ocr-finetune-pretrained.ckpt -P models/ocr-finetuned
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/hparams-ocr-finetune.yaml -P models/ocr-finetuned
find . -name __MACOSX -exec rm -rf {} \;
```

Inference:
```
python -m classification.inference --checkpoint=models/ocr-finetuned/ocr-finetune-pretrained.ckpt --hparams=models/ocr-finetuned/hparams-ocr-finetune.yaml --image_dir=resources/images/test_streetview
```

Available argparse parameter:
```
--checkpoint CHECKPOINT
    Checkpoint to already trained model (*.ckpt)
--hparams HPARAMS     
    Path to hparams file (*.yaml) generated during training
--image_dir IMAGE_DIR
    Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)
--gpu                 
    Use GPU for inference if CUDA is available, default to true
--batch_size BATCH_SIZE
--num_workers NUM_WORKERS
    Number of workers for image loading and pre-processing
```
The resulting file will be saved under the same directory as of the checkpoint file. 

#### Reproduce Results With Inference

To obtain the accuracy values for an inference model (in this example case, for ``GeoOCR finetuned``), run the following command:
```
python -m classification.test --checkpoint=models/ocr-finetuned/ocr-finetune-pretrained.ckpt --hparams=models/ocr-finetuned/hparams-ocr-finetune.yaml --image_dir=resources/images/test_streetview --ocr_json_dir=resources/test_feat_id.json --ocr_feat_dir=resources/test_ocr_feats_final.npy --label_dir=resources/test_streetview_labels.json
```

Available argparse paramters:
```
--checkpoint CHECKPOINT
    Checkpoint to already trained model (*.ckpt)
--hparams HPARAMS     
    Path to hparams file (*.yaml) generated during training
--image_dir IMAGE_DIR 
    Image folder to evaluate
--ocr_json_dir
    The .json file that keeps the indexing of the ocr features for a given file name (coordinates)
--ocr_feat_dir
    The .npy file containing the ocr features for a given index
--label_dir
    The .json file containing the class labels per hierarchical level for a given file name 
--gpu
    Use GPU for inference if CUDA is available, default to True
--precision PRECISION
    Full precision (32), half precision (16)
--batch_size BATCH_SIZE
--num_workers NUM_WORKERS
    Number of workers for image loading and pre-processing
```
The resulting file will be saved under the same directory as of the checkpoint file. 

### Training from Scratch
#### Train from scratch by using the current dataset
If you did not run the commands for the inference step, run them:
```
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_streetview_labels.json -O resources/test_streetview_labels.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_ocr_feats_final.npy -O resources/test_ocr_feats_final.npy
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_feat_id.json -O resources/test_feat_id.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/s2_cells.zip -P resources/
unzip resources/s2_cells.zip -d resources/
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/test_streetview.zip -P resources/images
unzip resources/images/test_streetview.zip -d resources/images/test_streetview/
```
Run the additional commands:
```
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/train_streetview_labels.json -O resources/train_streetview_labels.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/train_ocr_feats_final.npy -O resources/train_ocr_feats_final.npy
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/train_feat_id.json -O resources/train_feat_id.json
wget https://github.com/mhamzaerol/image-localization-utilizing-textual-information/releases/download/files/train_streetview.zip -P resources/images
unzip resources/images/train_streetview.zip -d resources/images/train_streetview/
find . -name __MACOSX -exec rm -rf {} \;
```
Train ``GeoOCR finetune`` from scratch by referring to the config corresponding file:
```
python -m classification.train_base --config config/ocr-finetune.yml --progbar 
```
The log files (loss/accuracy/checkpoints) will be populated under the ``data/experiments/`` directory.

#### Train from scratch by using an arbitrary dataset
Assuming you have a dataset (split into train and test) different than what is being used here. To train everything from scratch in this case, run the following commands. Here, we are assuming that you somehow placed the train and test datasets under the directory ``resources/images/test_streetview`` and ``resources/images/train_streetview``. Moreover, we are also assuming that the names of the images are the latitude,longitude coordinates of the locations the images were taken in. The name format should be "lat,lng.png" for an image. 
Firstly, based on the training set, we will create the partitionings. The characteristics of the partitionings will be determined by the config file provided as an input. Then, we will assign each of the images of the dataset (test/train) a label (namely a class index). 
```
python partitioning/create_cells.py --config=config/ocr-finetune.yml
python partitioning/assign_classes.py --config=config/ocr-finetune.yml
```
Then extract the OCR features:
```
python easy-ocr/main.py --config=config/ocr-finetune.yml
```
Then do the training:
```
python -m classification.train_base --config config/ocr-finetune.yml --progbar 
```

### Concluding remarks
- You can run the following train/test/inference commands for other models by changing the directories/config file names etc. appropriately.
