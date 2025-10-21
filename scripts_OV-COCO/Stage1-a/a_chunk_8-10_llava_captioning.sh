#!/bin/bash

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"


IMAGE_FOLDER="./datasets/coco/val2017"
IMAGE_ANNO_PATH="./datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json"

QS_FILE="./stage1_questions/list_all_objects.jsonl"
ANSWERS_FOLDER="./stage1_answers/coco_full80"
ANSWERS_FILE="answered_7b_annotations_coco_full80"

NUM_CHUNKS=10
CHUNK_IDX=8

GPU_IDS=$1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python run_stage1.py \
      --query-mode "captioning" \
      --dataset-name "coco_full80" \
      --model-path "${MODEL_PATH}" \
      --image-folder "${IMAGE_FOLDER}" \
      --image-anno-path "${IMAGE_ANNO_PATH}" \
      --question-file "${QS_FILE}" \
      --answers-folder "${ANSWERS_FOLDER}" \
      --answers-file "${ANSWERS_FILE}" \
      --num-chunks ${NUM_CHUNKS} \
      --chunk-idx ${CHUNK_IDX} \
      --conv-adapt