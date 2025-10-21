# Written by Mingxuan Liu
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from copy import deepcopy

from my_agents.fileios import *

# VLM agents
from my_agents.vision.llava.bot import LLaVA
from my_agents.vision.llava.utils import disable_torch_init

# Embedding agents
from my_agents.embedding.clip import CLIP
from my_agents.embedding.sbert import SBERT
# from my_agents.embedding.word2vec import SpacyEmbedding
from my_agents.embedding.embedding3 import Embedding3Model
from my_agents.embedding.utils import dot_score1, fuse_embeddings, tfms

from PIL import Image
import math

import spacy

# LLM agents
from my_agents.language.gpts.gpts import GPTS

from datasets.dataset_cnames_v3_synonyms import class_names_v3


def add_synonyms(args):
    def postprocess_proposal(raw_proposal):
        # formatting the answers
        answers = raw_proposal.split("*")
        results = []
        for answ in answers:
            answ = answ.strip()
            if 2 <= len(answ) <= 50:
                results.append(answ.lower())
        return list(set(results))

    # create Embedding model to compute similarities --llm-prompt-proposing
    agent_llm = GPTS(model='gpt-3.5-turbo-0125', temperature=args.llm_temperature)

    # given Full GT vocabulary
    full_vocabulary = class_names_v3[args.dataset_name]

    template_synonyms = """
As a machine learning researcher, I'm investigating how vision-language models interpret and describe various categories.

Specifically, I'm interested in uncovering the synonyms a model might use for the category '[__CATEGORY__]'.
Your job is to assist me by listing common synonyms for the category '[__CATEGORY__]' Please format your response with bullet points "*" for each synonym.

Here’s an example for the category 'tv':
* television
* televisions
* TV
* telly

Now, please provide a list of synonyms for '[__CATEGORY__]' in the same bullet point "*" format. Your response:
    """
    for cname, cinfo in tqdm(full_vocabulary.items()):
        this_prompt = template_synonyms.replace("[__CATEGORY__]",
                                                cname.replace("_", " "))
        synonyms_results = []

        # query three times
        for i in range(3):
            this_reply = agent_llm.infer(this_prompt)
            this_reply = postprocess_proposal(this_reply)
            print(this_reply)
            synonyms_results.extend(this_reply)

        # always remember to de-duplicate
        cinfo['synonyms'].extend(list(set(synonyms_results)))

    # Construct the new file name with replacements and formatting
    destination_path = os.path.join(args.answers_folder, args.answers_file.replace(
        "answered_", f"synonyms_{args.llm_model_name}_"
    ))

    # Dump the JSON data to the constructed path
    dump_json(destination_path, full_vocabulary)
    print("Successfully saved synonyms to {}".format(destination_path))
