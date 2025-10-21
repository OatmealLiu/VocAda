#!/bin/bash

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"

IMAGE_FOLDER="./datasets/coco/val2017"
IMAGE_ANNO_PATH="./datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json"

QS_FILE="./stage1_questions/list_all_objects.jsonl"
ANSWERS_FOLDER="./stage1_answers/coco_full80"
ANSWERS_FILE="answered_7b_annotations_coco_full80"

NUM_CHUNKS=10
CHUNK_IDX=1



CUDA_VISIBLE_DEVICES=2 python eval_stage1.py \
      --query-mode "proposing" \
      --dataset-name "coco_full80" \
      --pipeline-proposing "embedding" \
      --proposing-topk 3 \
      --proposing-thresh 0.05 \
      --proposing-alpha 0.5 \
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