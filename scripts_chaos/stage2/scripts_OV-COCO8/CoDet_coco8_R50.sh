#!/bin/bash

NUM_GPUS=$1

CFG_PATH="./configs_codet/CoDet_OVCOCO_R50_1x.yaml"
MODEL_PATH="./models/CoDet_OV-COCO/CoDet_OVCOCO_R50_1x.pth"

CLASSIFIER_NAME="coco_clip_a+cname.npy"

python train_net_codet.py \
        --num-gpus ${NUM_GPUS} \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('coco8_val',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('datasets/metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False
