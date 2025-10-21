#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-output/coco80_zeroshot_spotdet_V2_LLM_Img.out

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

CFG_PATH="./configs_detic/Detic_OVCOCO_CLIP_R50_1x_max-size.yaml"
MODEL_PATH="models/Detic_OV-COCO/Detic_OVCOCO_CLIP_R50_1x_max-size.pth"

CLASSIFIER_NAME="coco_clip_a+cname.npy"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('coco_generalized_zeroshot_val_spotdet_v2_llm',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False


