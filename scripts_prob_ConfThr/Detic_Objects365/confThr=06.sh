#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 48:00:00
#SBATCH --output=./slurm-output/ConfStudy_Obj365-SwinB_confThr=06

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

CFG_PATH="./configs_detic_ConfThr/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"
CONF_THR=0.6

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        --confThr ${CONF_THR} \
        DATASETS.TEST "('objects365_v2_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False

