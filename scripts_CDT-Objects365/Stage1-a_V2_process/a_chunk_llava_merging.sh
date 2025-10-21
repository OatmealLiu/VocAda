#!/bin/bash

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"


IMAGE_FOLDER="./datasets/objects365/val"
IMAGE_ANNO_PATH="./datasets/objects365/annotations/zhiyuan_objv2_val_fixed_everything.json"

QS_FILE="./stage1_questions/list_all_objects_v2.jsonl"
ANSWERS_FOLDER="./stage1_answers/objects365_V2"
ANSWERS_FILE="answered_V2_7b_annotations_objects365"

NUM_CHUNKS=100
CHUNK_IDX=20

python -m SpotDet.main \
      --query-mode "merging" \
      --dataset-name "objects365" \
      --model-path "${MODEL_PATH}" \
      --image-folder "${IMAGE_FOLDER}" \
      --image-anno-path "${IMAGE_ANNO_PATH}" \
      --question-file "${QS_FILE}" \
      --answers-folder "${ANSWERS_FOLDER}" \
      --answers-file "${ANSWERS_FILE}" \
      --num-chunks ${NUM_CHUNKS} \
      --chunk-idx ${CHUNK_IDX} \
      --conv-adapt