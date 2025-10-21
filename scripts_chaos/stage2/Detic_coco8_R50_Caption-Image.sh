#!/bin/bash

NUM_GPUS=$1

CFG_PATH="./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="coco_clip_a+cname.npy"

python train_net_detic.py \
        --num-gpus ${NUM_GPUS} \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('coco8_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False

