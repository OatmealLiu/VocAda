#!/bin/bash


CFG_PATH="./configs_detic/only_test_Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
MODEL_PATH="./models/Detic_CDT/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"

CLASSIFIER_NAME="tools_a+cname.npy"

CUDA_VISIBLE_DEVICES=0  python train_net_detic.py \
        --num-gpus 1 \
        --config-file ${CFG_PATH} \
        --eval-only \
        DATASETS.TEST "('tools_val_gt',)" \
        MODEL.WEIGHTS ${MODEL_PATH} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('metadata/${CLASSIFIER_NAME}',)" \
        MODEL.TEST_NUM_CLASSES "(84,)" \
        MODEL.MASK_ON False

[07/28 22:55:26 d2.evaluation.testing]: copypaste: Task: bbox
[07/28 22:55:26 d2.evaluation.testing]: copypaste: AP50
[07/28 22:55:26 d2.evaluation.testing]: copypaste: 0.3987