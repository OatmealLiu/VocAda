import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


LLAVA_MODEL_ZOO = {
    'llava-v1.6-mistral-7b': {
        'llava_size': '7b',
        'llava_base': 'mistral',
        'llava_version': 'v1.6',
        'llava_conv_mode': ['mistral_instruct', 'llava_v1'],
    },
    'llava-v1.6-34b': {
        'llava_size': '34b',
        'llava_base': 'nous-hermes-2-yi',
        'llava_version': 'v1.6',
        'llava_conv_mode': ['chatml_direct', 'llava_v1'],
    },
}


def ensure_folder_exists(folder_path):
    """Ensure that a folder exists. Create it if it does not."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def dump_json(filename, in_data):
    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fbj:
        if isinstance(in_data, dict):
            json.dump(in_data, fbj, indent=4)
        elif isinstance(in_data, list):
            json.dump(in_data, fbj)
        else:
            raise TypeError(f"in_data has wrong data type {type(in_data)}")


def load_json(filename):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_coco_format(args):
    # Model
    # Parse model type, size from the given model path name
    # Load the model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Choose the conversation mode (system-level prompt)
    args.conv_mode = LLAVA_MODEL_ZOO[model_name]["llava_conv_mode"][0 if args.conv_adapt else 1]

    # Load prompt templates
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # Split a list into n (roughly) equal-sized chunks
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    annotations = load_json(args.image_anno_path)
    # Create an img_idx list for iteration
    image_indices = list(range(len(annotations["images"])))

    # >>> go over each image >>>
    for this_image_idx in image_indices:
        # load the image
        this_image = Image.open(os.path.join(
                args.image_folder, annotations["images"][this_image_idx]["file_name"]
        )).convert('RGB')
        this_image_tensor = process_images([this_image], image_processor, model.config)[0]

        # >>> go over each prompt and collect the union of the answers >>>
        for qs_entry in tqdm(questions):    # Fetch a question
            # Parse a question entry
            qs_id = qs_entry["question_id"]
            qs_txt = qs_entry["text"]
            # Curate the question text a bit following LLaVA way
            cur_prompt = qs_txt
            if model.config.mm_use_im_start_end:
                qs_txt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_txt
            else:
                qs_txt = DEFAULT_IMAGE_TOKEN + '\n' + qs_txt

            # Initiate the conversation
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs_txt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()  # get the input txt prompt

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()

            # Actual LLaVA query inference
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=this_image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[this_image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            # Parse the output results
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            annotations["images"][this_image_idx].setdefault("answers_llava", []).append(
                {
                    "question_id": qs_id,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
        # <<< go over each prompt and collect the union of the answers <<<
    # <<< go over each image <<<
    dump_json(os.path.join(args.answers_folder, args.answers_file), annotations)


# LLaVA original
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # go over each question-image pairs (prompt-image-pairs)
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    """DEMO SCRIPT FROM REPO ROOT
    #!/bin/bash

    CHUNKS=8
    for IDX in {0..7}; do
        CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
            --model-path liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
            --question-file ~/haotian/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
            --image-folder ~/haotian/datasets/ScienceQA/data/scienceqa/images/test \
            --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --conv-mode llava_v1 &
    done
    """
    parser = argparse.ArgumentParser()
    # INPUTS
    # checkpoint path
    parser.add_argument("--model-path", type=str,
                        default="/home/mliu/GoldsGym/global_weights/llava-v1.6-mistral-7b")
    # if it's set to None, the CLIP model will be automatically downloaded from Hugging face
    parser.add_argument("--model-base", type=str,
                        default=None)
    # query image root folder
    parser.add_argument("--image-folder", type=str,
                        default="/home/mliu/GoldsGym/SpotDet/datasets/coco8/images")
    # query image annotation files in COCO format
    parser.add_argument("--image-anno-path", type=str,
                        default="/home/mliu/GoldsGym/SpotDet/datasets/coco8/annotations/annotations_coco8.json")

    # question (prompt) path
    parser.add_argument("--question-file", type=str,
                        default="/home/mliu/GoldsGym/SpotDet/stage1_questions/list_all_objects.jsonl")
    # conversation mode
    parser.add_argument("--conv-mode", type=str,
                        default="llava_v1")
    parser.add_argument('--conv-adapt', action='store_true', default=False,
                        help='weather adapt the conversation mode to the model type and size')

    # hyperparameters
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # When questions are few, just set it to num_chunks=1, chunk_idx = 0, so that all prompts will be processed in 1
    # batch
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    # OUTPUTS
    # queried answer files
    # the folder to put answers
    parser.add_argument("--answers-folder", type=str,
                        default="/home/mliu/GoldsGym/SpotDet/stage1_answers/coco8")
    # answer file name
    parser.add_argument("--answers-file", type=str,
                        default="answered_annotations_coco8")

    args = parser.parse_args()

    # makesure the output folder exists (will create one if not)
    ensure_folder_exists(args.answers_folder)

    # original LLaVA format
    # eval_model(args)

    # my COCO format for detection
    eval_coco_format(args)
