#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --output=./slurm-output/obj365_gpu/a_V2_chunk_%A_llava_objects365.out

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate spotdet

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"


IMAGE_FOLDER="./datasets/objects365/val"
IMAGE_ANNO_PATH="./datasets/objects365/annotations/zhiyuan_objv2_val_fixed_everything.json"

QS_FILE="./stage1_questions/list_all_objects_v2.jsonl"
ANSWERS_FOLDER="./stage1_answers/objects365_V2"
ANSWERS_FILE="answered_V2_7b_annotations_objects365"

NUM_CHUNKS=100

# Check if a chunk index is provided as a command-line argument
if [ -z "$1" ]; then
    echo "Error: No chunk index provided. Usage: sbatch this_script.sh <chunk_idx>"
    exit 1
else
    CHUNK_IDX=$1
fi

python -m SpotDet.main \
      --query-mode "captioning" \
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