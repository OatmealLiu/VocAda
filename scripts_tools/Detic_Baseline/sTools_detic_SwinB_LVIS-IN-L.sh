#!/bin/bash

CFG_PATH="./configs_detic/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="tools_a+cname.npy"

CUDA_VISIBLE_DEVICES=0  python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('tools_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(84,)" \
        MODEL.MASK_ON False