#!/bin/bash

ROOT_DIR="$HOME/source/ELDES_MESO/"
OUT_DIR=$ROOT_DIR/out/
DV_DIR=$ROOT_DIR/meso_example/
TRT_DIR=$DV_DIR

cd $ROOT_DIR
mkdir -p $OUT_DIR

python realtime_parallel.py --time '221790745' --codedir $ROOT_DIR --outdir $OUT_DIR --dvdir $DV_DIR --lomdir $TRT_DIR

python realtime_plot.py --time '221790745'  --codedir $ROOT_DIR --outdir $OUT_DIR --dvdir $DV_DIR --lomdir $TRT_DIR

# I can just run some plots without running the analysis (realtime_parallel.py)
# If I want restart the algorihtm, I should remove the files from ROT
# The IM are some other outputs (not correspond to ROT)
#0730 - 0900 every 5 minutes
