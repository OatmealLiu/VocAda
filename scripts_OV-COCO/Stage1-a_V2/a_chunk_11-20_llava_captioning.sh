#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-output/a_V2_chunk_11-20_llava_coco_full80.out

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"


IMAGE_FOLDER="./datasets/coco/val2017"
IMAGE_ANNO_PATH="./datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json"

QS_FILE="./stage1_questions/list_all_objects_v2.jsonl"
ANSWERS_FOLDER="./stage1_answers/coco_full80_V2"
ANSWERS_FILE="answered_V2_7b_annotations_coco_full80"

NUM_CHUNKS=20
CHUNK_IDX=11

python run_stage1.py \
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