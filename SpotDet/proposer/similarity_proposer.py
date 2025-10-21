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


def do_embedding_proposal(args):
    """
    Embedding-proposal v3
    include synonyms in the target CoI embeddings
    Given an image caption and a list of extracted noun chunks from the caption, we use embedding similarity scores to
    select class names from the given vocabulary (full class names of the dataset) using top-K method and thresholding.
    """
    # create NLP pipeline to extract nouns
    NLP_ZOO = {
        "small": "en_core_web_sm",
        "middle": "en_core_web_md",
        "large": "en_core_web_lg",
        "transformer": "en_core_web_trf",
    }
    nlp_model = spacy.load(NLP_ZOO[args.nlp_model_size])

    # create Embedding model to compute similarities
    if args.embedding_model_name == 'clip':
        agent_embedding = CLIP(model_size=args.embedding_model_size, device=args.device)
    elif args.embedding_model_name == 'embedding3':
        agent_embedding = Embedding3Model(model_size=args.embedding_model_size, device=args.device)
    else:  # default embedding model is sentence-bert
        agent_embedding = SBERT(model_size=args.embedding_model_size, device=args.device)


    # Load original annotation file
    gt_annos = load_json(args.image_anno_path)

    captioned_results_path = os.path.join(args.answers_folder,
                                          args.answers_file.replace("answered",
                                                                    f"answered_all_chunks{args.num_chunks}"))
    captioned_results = load_json(captioned_results_path)['images']

    # should have same number of image entries
    assert len(captioned_results) == len(gt_annos['images'])

    image_indexed_captioned_results = {this_caption_entry['id']: [cap['text']
                                                                  for cap in this_caption_entry['answers_llava']]
                                       for this_caption_entry in captioned_results}

    # given Full GT vocabulary
    full_vocabulary_dict_v3_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_{args.llm_model_name}_")))
    full_vocabulary_names = list(full_vocabulary_dict_v3_with_synonyms.keys())

    if args.use_synonyms is True:
        print("Using synonyms")
        e_vocabulary = []
        for cname, cinfo in full_vocabulary_dict_v3_with_synonyms.items():
            this_e_synonyms = agent_embedding.encode_text(cinfo['synonyms']+[cname])
            e_vocabulary.append(torch.mean(this_e_synonyms, dim=0))
        e_vocabulary = torch.stack(e_vocabulary)
        print(e_vocabulary.shape)
    else:
        e_vocabulary = agent_embedding.encode_text(full_vocabulary_names)

    # go over images
    for image_entry in tqdm(gt_annos['images']):
        image_id         = image_entry['id']
        captioned_result = image_indexed_captioned_results[image_id]

        # load the image
        if args.dataset_name == 'lvis':
            this_split_folder, this_jpg_name = image_entry["coco_url"].split("/")[-2:]
            image_file_name = os.path.join(
                args.image_folder, this_split_folder, this_jpg_name
            )
        else:
            image_file_name = os.path.join(args.image_folder, image_entry["file_name"])

        # container for CoI proposals
        coi_proposals   = []

        # visual context embedding
        if isinstance(agent_embedding, CLIP):
            # load the image
            this_image = Image.open(image_file_name).convert('RGB')
            this_image = tfms(this_image).unsqueeze(0).to(args.device)
            e_img = agent_embedding.encode_image(this_image)
        else:
            e_img = None

        # go over each caption
        for caption in captioned_result:
            # textual context embedding
            e_cap = agent_embedding.encode_text([caption], truncate=True) \
                    if isinstance(agent_embedding, CLIP) else agent_embedding.encode_text([caption])

            # run spaCy pipeline to extract nouns
            extracted_nouns = extract_nouns(nlp_model=nlp_model, caption=caption, root=False)
            extracted_nouns = deduplicate_list(extracted_nouns)

            e_query = agent_embedding.encode_text(extracted_nouns, truncate=True) \
                if isinstance(agent_embedding, CLIP) else agent_embedding.encode_text(extracted_nouns)

            # fuse the embeddings
            e_q_fused = fuse_embeddings(e_query, e_cap, e_img, alpha=args.proposing_alpha, beta=args.proposing_beta)

            # compute similarity scores
            mm_similarity = dot_score1(e_q_fused, e_vocabulary)

            # Here, instead of finding the best buddy CoIs, find top-k similar CoIs
            top_k_scores, top_k_indices = torch.topk(mm_similarity, k=args.proposing_topk, dim=1)

            # Collect all names that satisfies the requirements!
            for scores, indices in zip(top_k_scores, top_k_indices):
                for score, index in zip(scores, indices):
                    if score > args.proposing_thresh:
                        coi_proposals.append(full_vocabulary_names[index])

        # convert proposed CoI names to category_ids (1-indexed!)
        indices_coi_proposals = [full_vocabulary_dict_v3_with_synonyms[coi]['category_id'] for coi in coi_proposals]
        indices_coi_proposals = deduplicate_list(indices_coi_proposals)
        # record it
        image_entry['spotlight_cois'] = sorted(indices_coi_proposals)   # Miu: this is 1-indexed

    # Construct the new file name with replacements and formatting
    destination_path = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_{args.embedding_model_name}"
        f"_synonyms={args.use_synonyms}"
        f"_topk={args.proposing_topk}"
        f"_alpha={args.proposing_alpha}_beta={args.proposing_beta}"
        f".json"
    )

    # Dump the JSON data to the constructed path
    dump_json(destination_path, gt_annos)
    print("Successfully saved to {}".format(destination_path))


