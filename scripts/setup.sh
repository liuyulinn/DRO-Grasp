#!/bin/bash

### Download checkpoints
mkdir -p ckpt

cd ckpt
wget https://github.com/zhenyuwei2003/DRO-Grasp/releases/download/v1.0/ckpt.zip
unzip ckpt.zip
rm ckpt.zip
cd ..

echo "Download checkpoint models finished!"


### Download data
mkdir -p "data/"

### if gdown not installed, exit with error 
command -v gdown >/dev/null 2>&1 || { echo >&2 "gdown is required but it's not installed. Aborting."; exit 1; }


gdown --id "1aucOvtVj22HZA6-aUvejI1OR5wFSpOP4" -O "data/data_urdf.zip"
unzip -q "data/data_urdf.zip" -d "data/" # -q for quiet mode

rm "data/data_urdf.zip"

gdown --id "1P7Vgpsqml3OMH5p8JhsjtxQ7Dl5I7Ax6" -O "data/PointCloud.zip"
unzip -q "data/PointCloud.zip" -d "data/" # -q for quiet mode

rm "data/PointCloud.zip"

echo "Download data finished!"
# unzip data.zip -d data
