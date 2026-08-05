#!/bin/bash

# Command line arguments
CKPT="exp/all_cats_exp/000/checkpoints/0001000.pt"
EXP_DIR="exp"
EXP_NAME="all_cats_exp_eval"

DATA_DIR="./data_src/shapenet/shapenetv2.1"
DATASET_NAME="shapenet"
CATS="all"
SPLITS=("test")

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
MASK_MODE="mask"

# evaluation params
MODE="--iou"       # one of: --recon, --mesh, --pointcloud, --iou, --cd
DENSITY=64
NCHUNKS=8
MCUBETH=0.5

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd $SCRIPT_DIR

python "$SCRIPT_DIR/eval_seg.py" \
    --ckpt $CKPT \
    --exp-dir $EXP_DIR \
    --exp-name $EXP_NAME \
    --data-dir $DATA_DIR \
    --dataset-name $DATASET_NAME \
    --cats $CATS \
    --splits "${SPLITS[@]}" \
    $MODE \
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
    --mask-mode $MASK_MODE \
    --density $DENSITY \
    --nchunks $NCHUNKS \
    --mcubeth $MCUBETH
