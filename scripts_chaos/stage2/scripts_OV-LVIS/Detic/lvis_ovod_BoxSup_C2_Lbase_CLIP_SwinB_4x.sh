#!/bin/bash

NUM_GPUS=$1


CFG_PATH="./configs_detic/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.yaml"
MODEL_PATH="models/Detic_OV-LVIS/BoxSup-C2_Lbase_CLIP_SwinB_896b32_4x.pth"

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
