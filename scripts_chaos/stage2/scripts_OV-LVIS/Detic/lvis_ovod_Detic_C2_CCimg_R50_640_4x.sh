#!/bin/bash

NUM_GPUS=$1


CFG_PATH="./configs_detic/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml"
MODEL_PATH="models/Detic_OV-LVIS/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="lvis_v1_clip_a+cname.npy"

python train_net_detic.py \
        --num-gpus ${NUM_GPUS} \
        --config-file ${CFG_PATH} \
        --eval-only \
        MODEL.WEIGHTS ${MODEL_PATH}
#        DATASETS.TEST "('lvis_v1_val',)" \
#        MODEL.RESET_CLS_TESTS True \
#        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
#        MODEL.TEST_NUM_CLASSES "(1203,)" \
#        MODEL.MASK_ON False
