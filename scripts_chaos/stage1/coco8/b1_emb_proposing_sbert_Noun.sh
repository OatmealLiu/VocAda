#!/bin/bash

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"

IMAGE_FOLDER="./datasets/coco8/images"
IMAGE_ANNO_PATH="./datasets/coco8/annotations/annotations_coco8.json"

QS_FILE="./stage1_questions/list_all_objects.jsonl"
ANSWERS_FOLDER="./stage1_answers/coco8"
ANSWERS_FILE="answered_7b_annotations_coco8.json"

NUM_CHUNKS=1
CHUNK_IDX=0

GPU_IDS=$1

CUDA_VISIBLE_DEVICES=${GPU_IDS} python run_stage1.py \
      --query-mode "proposing" \
      --dataset-name "coco8" \
      --pipeline-proposing "embedding" \
      --proposing-topk 3 \
      --proposing-thresh 0.05 \
      --proposing-alpha 0.0 \
      --proposing-beta 0.0 \
      --nlp-model-size "large" \
      --embedding-model-name "sbert" \
      --embedding-model-size "sbert_base" \
      --llm-prompt-proposing "stage1_questions/llm_prompt_proposing.txt" \
      --model-path "${MODEL_PATH}" \
      --image-folder "${IMAGE_FOLDER}" \
      --image-anno-path "${IMAGE_ANNO_PATH}" \
      --question-file "${QS_FILE}" \
      --answers-folder "${ANSWERS_FOLDER}" \
      --answers-file "${ANSWERS_FILE}" \
      --num-chunks ${NUM_CHUNKS} \
      --chunk-idx ${CHUNK_IDX} \
      --conv-adapt