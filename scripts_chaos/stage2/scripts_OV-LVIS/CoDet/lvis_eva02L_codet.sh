#!/bin/bash

NUM_GPUS=$1


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
