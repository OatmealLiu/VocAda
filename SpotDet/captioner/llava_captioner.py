# Written by Mingxuan Liu
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
from PIL import Image
import math
import spacy


# VLM agents
from my_agents.vision import LLaVA, BLIP2Beta
from my_agents.vision.llava.utils import disable_torch_init
# Embedding agents
from my_agents.embedding import CLIP, SBERT, Embedding3Model
# LLM agents
from my_agents.language import GPTS, LLaMA3sAlpha
# Taggers
from my_agents.tagging import RAMPP

# Utils
from my_agents.fileios import *
from my_agents.embedding.utils import dot_score1, fuse_embeddings, tfms
from my_agents.post_processers import deduplicate_list, postprocess_caption

from SpotDet.utils import extract_nouns

from datasets.dataset_cnames import class_names
from datasets.dataset_cnames_v2 import class_names_v2, coco_zeroshot_categories_all
from datasets.dataset_cnames_v3_synonyms import class_names_v3  # for synonyms


def query_captioning_large_scale(args):
    # Model
    # Parse model type, size from the given model path name
    # Load the model
    disable_torch_init()
    llava = LLaVA(
        args.model_path, args.model_base, args.conv_mode, conv_adapt=args.conv_adapt,
        temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
    )

    # Load prompt templates
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # Split a list into n (roughly) equal-sized chunks

    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    annotations = load_json(args.image_anno_path)
    image_entries = annotations['images']
    image_chunk = get_chunk(image_entries, args.num_chunks, args.chunk_idx-1)   # Miu: note, args.chunk_idx starts from 1

    print(66*"=")
    print("Captioning {} / {} chunk: {} images".format(args.chunk_idx, args.num_chunks, len(image_chunk)))
    print(66*"=")

    for entry in tqdm(image_chunk):
        # load the image
        if args.dataset_name == 'lvis':
            this_split_folder, this_jpg_name = entry["coco_url"].split("/")[-2:]
            this_image_file_name = os.path.join(
                args.image_folder, this_split_folder, this_jpg_name
            )
        else:
            this_image_file_name = os.path.join(args.image_folder, entry["file_name"])

        this_image = Image.open(this_image_file_name).convert('RGB')

        # >>> go over each prompt and collect the union of the answers >>>
        for qs_entry in questions:  # Fetch a question
            # Parse a question entry
            qs_id = qs_entry["question_id"]
            qs_txt = qs_entry["text"]

            # LLaVA inference
            out_results = llava.do_vqa(this_image, qs_txt)
            out_results['question_id'] = qs_id

            print(60 * "-")
            print(out_results['text'])
            print(60 * "-")

            ans_id = shortuuid.uuid()
            entry.setdefault("answers_llava", []).append(out_results)

        # <<< go over each prompt and collect the union of the answers <<<
    # <<< go over each image <<<
    destination_path = os.path.join(args.answers_folder,
                                    args.answers_file.replace("answered",
                                                              f"answered_chunk{args.chunk_idx}-{args.num_chunks}"))
    final_results = {'images': image_chunk}
    dump_json(destination_path, final_results)
    print("Successfully saved to {}".format(destination_path))


def merge_captioning_large_scale(args):
    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    annotations = load_json(args.image_anno_path)
    num_images = len(annotations['images'])

    captioned_image_entries = []
    for chunk_idx in range(args.num_chunks):
        this_result_file_name = os.path.join(args.answers_folder,
                                             args.answers_file
                                             .replace("answered", f"answered_chunk{1+chunk_idx}-{args.num_chunks}"))

        this_chunked_result = load_json(this_result_file_name)
        for this_chunked_entry in this_chunked_result['images']:
            captioned_image_entries.append(this_chunked_entry)

    assert len(captioned_image_entries) == num_images

    print(66*"=")
    print("Merged into {} image entries".format(num_images))
    print(66*"=")

    # <<< go over each image <<<
    destination_path = os.path.join(args.answers_folder,
                                    args.answers_file.replace("answered",
                                                              f"answered_all_chunks{args.num_chunks}"))
    final_results = {'images': captioned_image_entries}
    dump_json(destination_path, final_results)
    print("Successfully saved merged results to {}".format(destination_path))