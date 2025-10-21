#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/CDT_Objects365_BoxSup_SpotDet_Noun+Caption.out


CFG_PATH="./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="./models/Detic_CDT/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth"

CLASSIFIER_NAME="o365_clip_a+cnamefix.npy"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('objects365_v2_val_spotdet_v2_clip_noun_cap',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(365,)" \
        MODEL.MASK_ON False
