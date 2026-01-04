#!/bin/bash

# Command line arguments
N_GPUS=1
N_NODES=1

EPOCHS=150
LR=0.0001
GLOBAL_BATCH_SIZE=4096
PTS_PER_SHAPE=4096
NPC=2048
EXP_DIR="exp"
EXP_NAME="all_cats_exp"
RANDOM_SEED=42

DATA_DIR="/localhome/ava40/Desktop/HiT/data_src/shapenet/shapenetv2.1"
DATASET_NAME="shapenet"
CATS="all"
SPLITS=("train")


N_PARTS=(8 16 32)
N_LEVELS=3
N_PLANES=32
PLANE_FDIM=8
PF_DIM=3
ZF_DIM=512
TF_DIM=4
EF_DIM=256
GF_DIM=256
MODEL_TYPE="hit-volume"

W_IM_SAMPLE=1.0
W_SAMPLE=1.0
W_EQ=0.001
W_BBX=0.01
W_CENTER=0.001
W_OVERLAP=0.01
W_BALANCE=0.01
W_CONTAIN=0.01
MASK_MODE="mask"

set -x 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd $SCRIPT_DIR

torchrun \
    --nnodes=$N_NODES \
    --nproc_per_node=$N_GPUS \
    "$SCRIPT_DIR/train.py" \
    --epochs $EPOCHS \
    --lr $LR \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --pts-per-shape $PTS_PER_SHAPE \
    --npc $NPC \
    --exp-dir $EXP_DIR \
    --exp-name $EXP_NAME \
    --random-seed $RANDOM_SEED \
    --data-dir $DATA_DIR \
    --dataset-name $DATASET_NAME \
    --cats $CATS \
    --splits "${SPLITS[@]}" \
    --train \
    --nparts "${N_PARTS[@]}" \
    --nlevels $N_LEVELS \
    --nplanes $N_PLANES \
    --plane-fdim $PLANE_FDIM \
    --pf-dim $PF_DIM \
    --zf-dim $ZF_DIM \
    --tf-dim $TF_DIM \
    --ef-dim $EF_DIM \
    --gf-dim $GF_DIM \
    --model-type $MODEL_TYPE \
    --w-im-sample-loss $W_IM_SAMPLE \
    --w-sample-loss $W_SAMPLE \
    --w-equilibrium-loss $W_EQ \
    --w-bbx-loss $W_BBX \
    --w-center-loss $W_CENTER \
    --w-overlap-loss $W_OVERLAP \
    --w-balance-loss $W_BALANCE \
    --w-containment-loss $W_CONTAIN \
    --mask-mode $MASK_MODE