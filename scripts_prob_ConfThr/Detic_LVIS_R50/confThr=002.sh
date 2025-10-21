#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 48:00:00
#SBATCH --output=./slurm-output/ConfStudy_LVIS-R50_confThr=002

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

CFG_PATH="./configs_detic_ConfThr/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml"
MODEL_PATH="models/Detic_OV-LVIS/Detic_LbaseI_CLIP_R5021k_640b64_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="lvis_v1_clip_a+cname.npy"
CONF_THR=0.02

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        --confThr ${CONF_THR} \
        MODEL.WEIGHTS ${MODEL_PATH} \
        DATASETS.TEST "('lvis_v1_val_baseline',)" \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(1203,)" \
        MODEL.MASK_ON False

