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


def do_tagging_proposal(args):
    """
    Tagging-proposal v3
    include synonyms in the target CoI embeddings
    Given an image caption and a list of extracted noun chunks from the caption, we use embedding similarity scores to
    select class names from the given vocabulary (full class names of the dataset) using top-K method and thresholding.
    """

    # Load original annotation file
    gt_annos_for_tagging = load_json(args.image_anno_path)

    # given Full GT vocabulary
    full_vocabulary_dict_v3_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_{args.llm_model_name}_")))
    full_vocabulary_names = list(full_vocabulary_dict_v3_with_synonyms.keys())

    tags_openset = load_json(args.tags_file)

    # create Embedding model to compute similarities
    agent_embedding = CLIP(model_size="ViT-L/14", device=args.device)

    agent_closeset_tagger = RAMPP(
        model_path=args.pretrained_rampp_path,
        device=args.device,
    )

    agent_openset_tagger = RAMPP(
        model_path=args.pretrained_rampp_path,
        given_tags=tags_openset,
        device=args.device,
    )

    # Tagging images
    # go over images
    for image_entry in tqdm(gt_annos_for_tagging['images']):
        # load the image
        if args.dataset_name == 'lvis':
            this_split_folder, this_jpg_name = image_entry["coco_url"].split("/")[-2:]
            image_file_name = os.path.join(
                args.image_folder, this_split_folder, this_jpg_name
            )
        else:
            image_file_name = os.path.join(args.image_folder, image_entry["file_name"])

        this_image = Image.open(image_file_name).convert('RGB')
        output_closeset_tags = agent_closeset_tagger.assign_tags(this_image)
        output_openset_tags  = agent_openset_tagger.assign_tags(this_image)

        print("---> Close Set:\t{}".format(output_closeset_tags))
        print("---> Open  Set:\t{}".format(output_openset_tags))

        image_entry['tags_closeset'] = output_closeset_tags
        image_entry['tags_openset']  = output_openset_tags


    if args.use_synonyms:
        print("Using synonyms")
        e_vocabulary = []
        for cname, cinfo in full_vocabulary_dict_v3_with_synonyms.items():
            this_e_synonyms = agent_embedding.encode_text(cinfo['synonyms']+[cname])
            e_vocabulary.append(torch.mean(this_e_synonyms, dim=0))
        e_vocabulary = torch.stack(e_vocabulary)
        print(e_vocabulary.shape)
    else:
        e_vocabulary = agent_embedding.encode_text(full_vocabulary_names)

    # Generate CloseSet Tagging CoI proposals
    gt_annos_close_set = load_json(args.image_anno_path)
    # go over images
    for image_entry, tagged_entry in tqdm(zip(gt_annos_close_set['images'], gt_annos_for_tagging['images'])):
        # container for CoI proposals
        coi_proposals   = []

        closeset_tags = deduplicate_list(tagged_entry['tags_closeset'])

        if len(closeset_tags) > 0:
            e_query = agent_embedding.encode_text(closeset_tags, truncate=True)

            # compute similarity scores
            mm_similarity = dot_score1(e_query, e_vocabulary)

            # Here, instead of finding the best buddy CoIs, find top-k similar CoIs
            top_k_scores, top_k_indices = torch.topk(mm_similarity, k=1, dim=1)

            # Collect all names that satisfies the requirements!
            for scores, indices in zip(top_k_scores, top_k_indices):
                for score, index in zip(scores, indices):
                    coi_proposals.append(full_vocabulary_names[index])

            # convert proposed CoI names to category_ids (1-indexed!)
            indices_coi_proposals = [full_vocabulary_dict_v3_with_synonyms[coi]['category_id'] for coi in coi_proposals]
            indices_coi_proposals = deduplicate_list(indices_coi_proposals)
        else:
            indices_coi_proposals = [v['category_id'] for v in full_vocabulary_dict_v3_with_synonyms.values()]

        # record it
        image_entry['spotlight_cois'] = sorted(indices_coi_proposals)   # Miu: this is 1-indexed

    # Construct the new file name with replacements and formatting
    destination_path_close_set = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_tagging"
        f"_closeset"
        f"_synonyms={args.use_synonyms}"
        f".json"
    )

    # Dump the JSON data to the constructed path
    dump_json(destination_path_close_set, gt_annos_close_set)
    print("Successfully saved to {}".format(destination_path_close_set))

    # Generate OpenSet Tagging CoI proposals
    gt_annos_open_set = load_json(args.image_anno_path)
    # go over images
    for image_entry, tagged_entry in tqdm(zip(gt_annos_open_set['images'], gt_annos_for_tagging['images'])):
        coi_proposals = deduplicate_list(tagged_entry['tags_openset'])

        if len(coi_proposals) > 0:
            # convert proposed CoI names to category_ids (1-indexed!)
            indices_coi_proposals = [full_vocabulary_dict_v3_with_synonyms[coi]['category_id'] for coi in coi_proposals]
            indices_coi_proposals = deduplicate_list(indices_coi_proposals)
        else:
            indices_coi_proposals = [v['category_id'] for v in full_vocabulary_dict_v3_with_synonyms.values()]

        # record it
        image_entry['spotlight_cois'] = sorted(indices_coi_proposals)   # Miu: this is 1-indexed

    destination_path_open_set = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_tagging"
        f"_openset"
        f"_synonyms={args.use_synonyms}"
        f".json"
    )

    # Dump the JSON data to the constructed path
    dump_json(destination_path_open_set, gt_annos_open_set)
    print("Successfully saved to {}".format(destination_path_open_set))


