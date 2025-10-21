#!/bin/bash

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"

TAGGING_MODEL_PATH="/beegfs/scratch/project/clara/tasris/global_weights/RAM_models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth"

IMAGE_FOLDER="./datasets/coco/val2017"
IMAGE_ANNO_PATH="./datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json"

QS_FILE="./stage1_questions/list_all_objects_v2.jsonl"
ANSWERS_FOLDER="./stage1_answers/coco_full80_V2"
ANSWERS_FILE="answered_V2_7b_annotations_coco_full80"

NUM_CHUNKS=20
CHUNK_IDX=1



CUDA_VISIBLE_DEVICES=1 python eval_stage1.py \
      --query-mode "proposing" \
      --dataset-name "coco_full80" \
      --pipeline-proposing "tagging" \
      --tags-file "datasets/tags/coco_rampp_tags_openset.json" \
      --pretrained-rampp-path "${TAGGING_MODEL_PATH}" \
      --proposing-topk 1 \
      --proposing-thresh 0.01 \
      --proposing-alpha 0.0 \
      --proposing-beta 0.0 \
      --nlp-model-size "large" \
      --embedding-model-name "clip" \
      --embedding-model-size "ViT-L/14" \
      --llm-prompt-proposing "stage1_questions/llm_prompt_proposing_v2.txt" \
      --model-path "${MODEL_PATH}" \
      --image-folder "${IMAGE_FOLDER}" \
      --image-anno-path "${IMAGE_ANNO_PATH}" \
      --question-file "${QS_FILE}" \
      --answers-folder "${ANSWERS_FOLDER}" \
      --answers-file "${ANSWERS_FILE}" \
      --num-chunks ${NUM_CHUNKS} \
      --chunk-idx ${CHUNK_IDX} \
      --conv-adapt \
      --use-synonyms True