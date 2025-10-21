#!/bin/bash

NUM_GPUS=$1

CFG_PATH="./configs_detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.yaml"
MODEL_PATH="models/Detic_OV-COCO/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth"

CLASSIFIER_NAME="coco_clip_a+cname.npy"

# This is the best model

python train_net_detic.py \
        --num-gpus ${NUM_GPUS} \
        --config-file ${CFG_PATH} \
        --eval-only \
        MODEL.WEIGHTS ${MODEL_PATH}
#        MODEL.RESET_CLS_TESTS True \
#        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
#        MODEL.TEST_NUM_CLASSES "(80,)" \
#        MODEL.MASK_ON False
