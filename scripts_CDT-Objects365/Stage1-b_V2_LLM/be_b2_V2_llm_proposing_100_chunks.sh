#!/bin/bash
#SBATCH -p gpu-be
#SBATCH --gres gpu:2
#SBATCH --mem=64000
#SBATCH --time=120:00:00
#SBATCH --output=./slurm-output/obj365_gpu/gpu_be_b_V2_proposing_chunk_%A_llama3_objects365.out

MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-mistral-7b"
#MODEL_PATH="/gfs-ssd/project/clara/lbe-expts/checkpoints/llava/llava-v1.6-34b"
LLM_MODEL_NAME="llama3-8b-instruct"

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
      --wandb-mode "online" \
      --query-mode "proposing" \
      --dataset-name "objects365" \
      --pipeline-proposing "llm" \
      --llm-model-name "${LLM_MODEL_NAME}" \
      --proposing-topk 1 \
      --proposing-thresh 0.01 \
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
      --conv-adapt
