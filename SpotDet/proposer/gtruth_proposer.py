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


def do_gt_proposal(args):
    """
    Use ground-truth object categories as the extracted nouns
    This is considered as the Upper-bound case
    """
    # Load original annotation file
    gt_annos = load_json(args.image_anno_path)

    if 'coco' in args.dataset_name:
        """
        coco-class ID to classifier-ID mapping
        this is because COCO class-ID is not within [1, 80], it's between [1,90]
        """
        mapper_coco_ids = {}
        for i, cat_entry in enumerate(coco_zeroshot_categories_all):
            mapper_coco_ids[cat_entry['id']] = 1 + i

        for image_entry in tqdm(gt_annos['images']):
            # collect gt present object categories from annotations
            gt_coi_proposal = [anno_entry['category_id']
                               for anno_entry in gt_annos['annotations']
                               if anno_entry['image_id'] == image_entry['id']]

            gt_coi_proposal = [mapper_coco_ids[coi] for coi in gt_coi_proposal]
            # record
            image_entry['spotlight_cois'] = sorted(list(set(gt_coi_proposal)))
    else:
        """
        For the other datasets, where class-ID = classifier vector-ID, # of classes = classifier-dim
        """
        for image_entry in tqdm(gt_annos['images']):
            # collect gt present object categories from annotations
            gt_coi_proposal = [anno_entry['category_id']
                               for anno_entry in gt_annos['annotations']
                               if anno_entry['image_id'] == image_entry['id']]

            # record
            image_entry['spotlight_cois'] = sorted(list(set(gt_coi_proposal)))
            print(image_entry['spotlight_cois'])

    # Construct the new file name with replacements and formatting
    destination_path = args.image_anno_path.replace(".json", "_spotdet_gt.json")

    # Dump the JSON data to the constructed path
    dump_json(destination_path, gt_annos)
    # print("Successfully saved to {}".format(destination_path))
