#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 08:00:00
#SBATCH --output=./slurm-output/CoDet_OV-LVIS-EVA02-L.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

NUM_GPUS=${SLURM_GPUS:-8}

CFG_PATH="./configs_codet/CoDet_OVLVIS_EVA_4x.yaml"
MODEL_PATH="./models/CoDet_OV-LVIS/CoDet_OVLVIS_EVA_4x.pth"

CLASSIFIER_NAME="lvis_v1_clip_a+cname.npy"

python train_net_codet.py \
        --num-gpus ${NUM_GPUS} \
        --config-file ${CFG_PATH} \
        --eval-only \
        MODEL.WEIGHTS ${MODEL_PATH}
#        MODEL.RESET_CLS_TESTS True \
#        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
#        MODEL.TEST_NUM_CLASSES "(80,)" \
#        MODEL.MASK_ON False
