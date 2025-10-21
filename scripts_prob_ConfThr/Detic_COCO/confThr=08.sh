#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 48:00:00
#SBATCH --output=./slurm-output/ConfStudy_COCO_confThr=08

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

CFG_PATH="./configs_detic_ConfThr/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.yaml"
MODEL_PATH="models/Detic_OV-COCO/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"

CLASSIFIER_NAME="coco_clip_a+cname.npy"
CONF_THR=0.8

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        --confThr ${CONF_THR} \
        DATASETS.TEST "('coco_generalized_zeroshot_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False

