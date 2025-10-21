# Written by Mingxuan Liu
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re

# VLM agents
from my_agents.vision.llava.bot import LLaVA
from my_agents.vision.llava.utils import disable_torch_init

# Embedding agents
from my_agents.embedding.clip import CLIP
from my_agents.embedding.sbert import SBERT
# from my_agents.embedding.word2vec import SpacyEmbedding
from my_agents.embedding.embedding3 import Embedding3Model

# LLM agents
from my_agents.language.gpts.gpts import GPTS

from datasets.dataset_cnames import class_names
from datasets.dataset_cnames_v2 import class_names_v2, coco_zeroshot_categories_all
from datasets.dataset_cnames_v3_synonyms import class_names_v3  # for synonyms

from my_agents.embedding.utils import dot_score1, fuse_embeddings, tfms
from my_agents.tagging.bot import RAMPP

from my_agents.fileios import *
from my_agents.post_processers import deduplicate_list, postprocess_caption

from PIL import Image
import math

import spacy














def query_proposing_v3(args):
    if args.pipeline_proposing == 'ground_truth':
        do_gt_proposal(args)
    elif args.pipeline_proposing == 'llm':
        do_llm_proposal(args)
    elif args.pipeline_proposing == 'tagging':
        do_tagging_proposal(args)
    else:
        do_embedding_proposal(args)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str,
                        default="coco8",
                        choices=["coco8", "coco", "fsod_toy200",
                                 "lvis", "oid", "objects365", "v3det",
                                 "coco_full80", "fsod_full200"])
    parser.add_argument("--query-mode", type=str,
                        default="captioning",
                        choices=["captioning", "merging", "proposing", "merging_llm_proposals"])
    parser.add_argument("--pipeline-proposing", type=str,
                        default='embedding',
                        choices=["llm", "embedding", "ground_truth", "tagging"])

    # Tagging
    parser.add_argument("--tags-file", type=str, default="datasets/tags/coco_rampp_tags_openset.json")
    parser.add_argument('--pretrained-rampp-path',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='/beegfs/scratch/project/clara/tasris/global_weights/RAM_models/recognize-anything-plus-model/ram_plus_swin_large_14m.pth')
    # Similarity threshold for embedding-based proposer proposing_topk
    parser.add_argument("--proposing-topk", type=int, default=3)
    parser.add_argument("--proposing-thresh", type=float, default=0.15)
    parser.add_argument("--proposing-alpha", type=float, default=0.0, help="alpha: weight for caption context e_cap")
    parser.add_argument("--proposing-beta", type=float, default=0.0, help="beta: weight for image context e_img")
    parser.add_argument('--use-synonyms', default=True)
    parser.add_argument("--nlp-model-size", type=str,
                        default='large',
                        choices=["small", "middle", "large", "transformer"])
    parser.add_argument("--embedding-model-name", type=str,
                        default='sbert',
                        choices=["clip", "sbert", "embedding3"])
    parser.add_argument("--embedding-model-size", type=str,
                        default='sbert_base',
                        choices=[
                            # CLIPs
                            "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336",
                            # OpenAI embedding3s
                            "emb3_small", "emb3_large",
                            # Sentence-BERTs
                            "sbert_mini", "sbert_base", "sbert_search",
                        ])
    parser.add_argument("--llm-model-name", type=str,
                        default='gpt-3.5-turbo-0125',
                        choices=[
                            "gpt-3.5-turbo",
                            "gpt-3.5-turbo-0125",
                            "gpt-4",
                            "gpt-4-turbo-preview",
                            "gpt-4-0125-preview",
                        ])
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-prompt-proposing", type=str,
                        default="stage1_questions/llm_prompt_proposing.txt")
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
    parser.add_argument("--chunk-idx", type=int, default=1, help="1-index of the chunk")
    # OUTPUTS
    # queried answer files
    # the folder to put answers
    parser.add_argument("--answers-folder", type=str,
                        default="/home/mliu/GoldsGym/SpotDet/stage1_answers/coco8")
    # answer file name
    parser.add_argument("--answers-file", type=str,
                        default="answered_annotations_coco8")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device

    print(args)

    # makesure the output folder exists (will create one if not)
    ensure_folder_exists(args.answers_folder)

    if args.query_mode == "captioning":
        # my COCO format for detection
        query_captioning_large_scale(args)
    elif args.query_mode == "merging":
        merge_captioning_large_scale(args)
    elif args.query_mode == "proposing":
        query_proposing_v3(args)
    elif args.query_mode == "merging_llm_proposals":
        merge_llm_proposal_large_scale(args)
    else:
        raise NotImplementedError("{} is not implemented yet by Miu loll".format(args.query_mode))
