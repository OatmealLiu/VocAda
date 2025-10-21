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
import wandb

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

from SpotDet.utils import extract_nouns, clrprint

from datasets.dataset_cnames import class_names
from datasets.dataset_cnames_v2 import class_names_v2, coco_zeroshot_categories_all
from datasets.dataset_cnames_v3_synonyms import class_names_v3  # for synonyms


def format_class_prompt_with_synonyms(dict_class_synonyms):
    prompt = ""
    for index, (key, values) in enumerate(dict_class_synonyms.items(), start=1):
        synonyms = ', '.join([f'"{value}"' for value in values])
        prompt += f'\n{index}. "{key}": [{synonyms}]'
    return prompt


def do_llm_proposal(args, wandb_run, debug_knob=False):
    def postprocess_proposal(raw_proposal):
        # formatting the answers
        parenthese_pattern = r'\s*\([^)]*\)'

        answers = raw_proposal.split("*")
        results = []
        for answ in answers:
            answ = re.sub(parenthese_pattern, '', answ)
            answ = answ.strip()
            answ = answ.replace('"', '').replace("'", "").replace('\n', '').strip()
            if 3 <= len(answ) <= 50:
                results.append(answ.lower())
        return list(set(results))

    # create NLP pipeline to extract nouns
    NLP_ZOO = {
        "small": "en_core_web_sm",
        "middle": "en_core_web_md",
        "large": "en_core_web_lg",
        "transformer": "en_core_web_trf",
    }

    nlp_model = spacy.load(NLP_ZOO[args.nlp_model_size])

    if 'gpt' in args.llm_model_name:
        agent_llm = GPTS(model="gpt-3.5-turbo", temperature=args.temperature_llm)
    else:
        agent_llm = LLaMA3sAlpha(model=args.llm_model_name, temperature=0.99, max_new_tokens=512)

    # load corresponding captioning results
    captioned_results_path = os.path.join(
        args.answers_folder, args.answers_file.replace("answered", f"answered_chunk{args.chunk_idx}-{args.num_chunks}")
    )

    captioned_results = load_json(captioned_results_path)['images']

    # load llm prompt
    template_llm_prompt = load_txt(args.llm_prompt_proposing)

    full_vocabulary_dict_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_gpt-3.5-turbo-0125_")
    ))

    full_class_names = list(full_vocabulary_dict_with_synonyms.keys())

    if args.dataset_name == 'lvis':
        num_chunks_class_names = 20
    elif args.dataset_name == 'objects365':
        num_chunks_class_names = 6
    else:
        num_chunks_class_names = 1

    chunked_class_names = [get_chunk(full_class_names, num_chunks_class_names, i)
                           for i in range(num_chunks_class_names)]

    num_img_index = 0
    for this_img_entry in tqdm(captioned_results):
        # if we did it, we skip
        num_img_index += 1
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%% {num_img_index} / {len(captioned_results)} ")
        if len(this_img_entry.get('vanilla_proposals', [])) > .1:
            continue

        # container for CoI proposals
        this_img_vanilla_proposals = []

        for this_caption_entry in this_img_entry['answers_llava']:
            # prepare caption related prompts
            caption_text           = postprocess_caption(text=this_caption_entry['text'].strip())
            extracted_nouns        = extract_nouns(nlp_model=nlp_model, caption=caption_text, root=False)
            extracted_nouns        = deduplicate_list(extracted_nouns)
            prompt_caption_text    = "{}".format(caption_text)
            prompt_extracted_nouns = "{}".format(", ".join(extracted_nouns))

            print(60 * "=")

            this_cap_raw_proposals     = []
            this_cap_vanilla_proposals = []
            for achunk_names in chunked_class_names:
                prompt_num_classes               = str(len(achunk_names))
                prompt_class_names               = ", ".join(achunk_names)
                prompt_class_names_with_synonyms = format_class_prompt_with_synonyms(
                    {k: full_vocabulary_dict_with_synonyms[k]['synonyms'] for k in achunk_names}
                )

                # infer
                this_user_prompt = (template_llm_prompt
                                    .replace("[__CAPTION__]",              prompt_caption_text)
                                    .replace("[__EXTRACTED_NOUNS__]",      prompt_extracted_nouns)
                                    .replace("[__CANDIDATES_WITH_SYNO__]", prompt_class_names_with_synonyms)
                                    .replace("[__NUM_CLASSES__]",          prompt_num_classes)
                                    .replace("[__CANDIDATES__]",           prompt_class_names))

                # clrprint('\n' + this_user_prompt + '\n', c='light_green')

                raw_reply = agent_llm.infer(this_user_prompt)

                print("******** Raw Reply:\n")
                clrprint(raw_reply, c='light_blue')

                vanilla_reply = postprocess_proposal(raw_reply)

                print("******** Processed Reply:\n")
                clrprint(vanilla_reply, c='light_blue')


                this_cap_raw_proposals.append(raw_reply)
                this_cap_vanilla_proposals.extend(vanilla_reply)

            print(60*"=")

            this_caption_entry['raw_proposals']     = this_cap_raw_proposals
            this_caption_entry['vanilla_proposals'] = deduplicate_list(this_cap_vanilla_proposals)

            this_img_vanilla_proposals.extend(this_cap_vanilla_proposals)

        this_img_entry['vanilla_proposals'] = deduplicate_list(this_img_vanilla_proposals)

        wandb_run.log({'progress_id': num_img_index})
        # Dump the JSON data to the constructed path

        if debug_knob is True:
            if num_img_index >= 1:
                break
        else:
            # proposing checkpoints
            dump_json(captioned_results_path, {'images': captioned_results})
            print("Successfully saved to {}".format(captioned_results_path))

    # Dump the JSON data to the constructed path
    dump_json(captioned_results_path.replace("answered_chunk", "proposed_chunk"),
              {'images': captioned_results})
    print("Successfully saved to {}".format(captioned_results_path.replace("answered_chunk", "proposed_chunk")))


def merge_llm_proposal_large_scale(args):
    """
    2024-04-04: new merging
    """
    # Function to remove content within parentheses
    def remove_parentheses(content_list):
        pattern = r'\s*\([^)]*\)'  # Matches anything in parentheses, including the parentheses
        new_list = [re.sub(pattern, '', item) for item in content_list]
        return [item.strip().lower() for item in new_list]

    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    gt_annos = load_json(args.image_anno_path)
    num_images = len(gt_annos['images'])

    # given Full GT vocabulary
    full_vocabulary_dict_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_{args.llm_model_name}_")))

    # accumulate CoI proposal entries
    proposed_image_entries = []
    for chunk_idx in range(args.num_chunks):
        this_result_file_name = os.path.join(args.answers_folder,
                                             args.answers_file
                                             .replace("answered", f"proposed_chunk{1+chunk_idx}-{args.num_chunks}"))

        this_chunked_proposing_result = load_json(this_result_file_name)['images']
        proposed_image_entries.extend(this_chunked_proposing_result)

    # Miu: key=image_id, value=vanilla_proposals and captions
    image_indexed_proposed_results = {}
    for this_img_entry in proposed_image_entries:
        this_img_id            = this_img_entry['id']
        this_vanilla_proposals = this_img_entry['vanilla_proposals']
        this_caption_texts     = [this_cap_entry['text'].lower() for this_cap_entry in this_img_entry['answers_llava']]
        image_indexed_proposed_results[this_img_id] = {
            'vanilla_proposal_names': this_vanilla_proposals,
            'captions':               this_caption_texts,
        }

    assert len(image_indexed_proposed_results) == num_images

    print(66*"=")
    print("Merging proposal into {} image entries".format(num_images))
    print(66*"=")

    # update annotations
    for this_img_entry in tqdm(gt_annos['images']):
        this_img_id = this_img_entry['id']

        # spotlight_coi container
        spotlight_coi_indices = []

        # First: check caption space for direct hits
        for cname, cinfo in full_vocabulary_dict_with_synonyms.items():
            this_category_id = cinfo['category_id']

            target_names_list = [cname.lower().replace('_', ' ')] + cinfo['synonyms']
            target_names_list = remove_parentheses(target_names_list)

            for target_name in target_names_list:
                caption_space = "\n\n".join(image_indexed_proposed_results[this_img_id]['captions'])
                if target_name in caption_space:
                    spotlight_coi_indices.append(this_category_id)
                    break

        # Second: check LLM proposed vanilla names
        for cname, cinfo in full_vocabulary_dict_with_synonyms.items():
            this_category_id = cinfo['category_id']

            target_names_list = [cname.lower().replace('_', ' ')] + cinfo['synonyms']
            # target_names_list = remove_parentheses(target_names_list)

            for vanilla_name in image_indexed_proposed_results[this_img_id]['vanilla_proposal_names']:
                # In-Word
                # for target_name in target_names_list:
                #     if (vanilla_name.lower() in target_name) or (remove_parentheses([vanilla_name])[0] in target_name):
                #         spotlight_coi_indices.append(this_category_id)
                #         break

                # In-List
                if (vanilla_name.lower() in target_names_list) or (remove_parentheses([vanilla_name])[0] in target_names_list):
                    spotlight_coi_indices.append(this_category_id)
                    break

        if len(spotlight_coi_indices) > 0:
            this_img_entry['spotlight_cois'] = list(set(spotlight_coi_indices))
        else:
            this_img_entry['spotlight_cois'] = [i + 1 for i in range(len(full_vocabulary_dict_with_synonyms))]
    # Miu: This part is adapted to the COCO-mistake, but it's also fine for other sets

    # <<< go over each image <<<
    destination_path = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_GPT35_inList.json")

    dump_json(destination_path, gt_annos)
    print("Successfully saved merged results to {}".format(destination_path))


def temp_merge_llm_proposal_large_scale(args):
    """
    2024-04-04: new merging
    """
    # Function to remove content within parentheses
    def remove_numbers(content_list):
        new_content = []
        for entry in content_list:
            new_content.append(entry.split('.')[-1].lower().strip())
        return new_content

    def remove_parentheses(content_list):
        pattern = r'\s*\([^)]*\)'  # Matches anything in parentheses, including the parentheses
        new_list = [re.sub(pattern, '', item) for item in content_list]
        return [item.strip().lower() for item in new_list]

    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    gt_annos = load_json(args.image_anno_path)
    num_images = len(gt_annos['images'])

    # given Full GT vocabulary
    full_vocabulary_dict_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_gpt-3.5-turbo-0125_")
    ))
    # accumulate CoI proposal entries
    proposed_image_entries = []
    for chunk_idx in range(args.num_chunks):
        this_result_file_name = os.path.join(args.answers_folder,
                                             args.answers_file
                                             .replace("answered", f"answered_chunk{1+chunk_idx}-{args.num_chunks}"))

        this_chunked_proposing_result = load_json(this_result_file_name)['images']
        proposed_image_entries.extend(this_chunked_proposing_result)

    # Miu: key=image_id, value=vanilla_proposals and captions
    image_indexed_proposed_results = {}
    for this_img_entry in proposed_image_entries:
        this_img_id            = this_img_entry['id']
        this_vanilla_proposals = this_img_entry.get('vanilla_proposals', [])
        this_caption_texts     = [this_cap_entry['text'].lower() for this_cap_entry in this_img_entry['answers_llava']]
        image_indexed_proposed_results[this_img_id] = {
            'vanilla_proposal_names': this_vanilla_proposals,
            'captions':               this_caption_texts,
        }

    assert len(image_indexed_proposed_results) == num_images

    print(66*"=")
    print("Merging proposal into {} image entries".format(num_images))
    print(66*"=")

    # update annotations
    for this_img_entry in tqdm(gt_annos['images']):
        this_img_id = this_img_entry['id']

        # not LLM-proposed yet
        if len(image_indexed_proposed_results[this_img_id]['vanilla_proposal_names']) <= .1:
            this_img_entry['spotlight_cois'] = [i + 1 for i in range(len(full_vocabulary_dict_with_synonyms))]
            continue

        # spotlight_coi container
        spotlight_coi_indices = []

        # First: check caption space for direct hits
        # for cname, cinfo in full_vocabulary_dict_with_synonyms.items():
        #     this_category_id = cinfo['category_id']
        #
        #     target_names_list = cinfo['synonyms']
        #     cname_list = cname.split('/')
        #     cname_list = [cn.replace('_', ' ').strip().lower() for cn in cname_list]
        #     target_names_list.extend(cname_list)
        #
        #     target_names_list = remove_parentheses(target_names_list)
        #     target_names_list = remove_numbers(target_names_list)
        #
        #     for target_name in target_names_list:
        #         caption_space = "\n\n".join(image_indexed_proposed_results[this_img_id]['captions'])
        #         if target_name in caption_space:
        #             spotlight_coi_indices.append(this_category_id)
        #             break

        # Second: check LLM proposed vanilla names
        for cname, cinfo in full_vocabulary_dict_with_synonyms.items():
            this_category_id = cinfo['category_id']

            target_names_list = cinfo['synonyms']
            cname_list = cname.split('/')
            cname_list = [cn.replace('_', ' ').strip().lower() for cn in cname_list]
            target_names_list.extend(cname_list)

            target_names_list = remove_parentheses(target_names_list)
            target_names_list = remove_numbers(target_names_list)

            target_proposed = False

            for target_name in target_names_list:
                caption_space = "\n\n".join(image_indexed_proposed_results[this_img_id]['captions'])
                if target_name in caption_space:
                    spotlight_coi_indices.append(this_category_id)
                    target_proposed = True
                    break

            if target_proposed is True:
                continue

            for vanilla_name in image_indexed_proposed_results[this_img_id]['vanilla_proposal_names']:
                vanilla_names = vanilla_name.split('/')
                vanilla_names = [vn.replace('_', ' ').strip().lower() for vn in vanilla_names]

                for this_vanilla_name in vanilla_names:
                    # # In-Word
                    # for target_name in target_names_list:
                    #     if (this_vanilla_name.lower() in target_name) or (remove_parentheses([this_vanilla_name])[0] in target_name):
                    #         spotlight_coi_indices.append(this_category_id)
                    #         target_proposed = True
                    #         break
                    # if target_proposed:
                    #     break

                    # In-List
                    if (this_vanilla_name.lower() in target_names_list) or (remove_parentheses([this_vanilla_name])[0] in target_names_list):
                        spotlight_coi_indices.append(this_category_id)
                        target_proposed = True
                        break

                if target_proposed is True:
                    break

        if len(spotlight_coi_indices) > 0:
            this_img_entry['spotlight_cois'] = list(set(spotlight_coi_indices))
        else:
            this_img_entry['spotlight_cois'] = [i + 1 for i in range(len(full_vocabulary_dict_with_synonyms))]
    # Miu: This part is adapted to the COCO-mistake, but it's also fine for other sets

    # <<< go over each image <<<
    destination_path = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_GPT35_inList.json")

    dump_json(destination_path, gt_annos)
    print("Successfully saved merged results to {}".format(destination_path))



def temp_merge_llm_proposal_with_clip(args):
    # Function to remove content within parentheses
    def remove_numbers(content_list):
        new_content = []
        for entry in content_list:
            new_content.append(entry.split('.')[-1].lower().strip())
        return new_content

    # Load image annotation files that contain image paths
    # Note: here we expect COCO-/LVIS-style JSON annotation file
    gt_annos = load_json(args.image_anno_path)
    num_images = len(gt_annos['images'])

    # given Full GT vocabulary
    full_vocabulary_dict_with_synonyms = load_json(os.path.join(
        args.answers_folder, args.answers_file.replace("answered_", f"synonyms_gpt-3.5-turbo-0125_")
    ))

    full_vocabulary_names = list(full_vocabulary_dict_with_synonyms.keys())

    # accumulate CoI proposal entries
    proposed_image_entries = []
    for chunk_idx in range(args.num_chunks):
        this_result_file_name = os.path.join(args.answers_folder,
                                             args.answers_file
                                             .replace("answered", f"answered_chunk{1+chunk_idx}-{args.num_chunks}"))

        this_chunked_proposing_result = load_json(this_result_file_name)['images']
        proposed_image_entries.extend(this_chunked_proposing_result)

    # Miu: key=image_id, value=vanilla_proposals and captions
    image_indexed_proposed_results = {}
    for this_img_entry in proposed_image_entries:
        this_img_id            = this_img_entry['id']
        this_vanilla_proposals = this_img_entry.get('vanilla_proposals', [])
        this_caption_texts     = [this_cap_entry['text'].lower() for this_cap_entry in this_img_entry['answers_llava']]
        image_indexed_proposed_results[this_img_id] = {
            'vanilla_proposal_names': this_vanilla_proposals,
            'captions':               this_caption_texts,
        }

    assert len(image_indexed_proposed_results) == num_images

    print(66*"=")
    print("Merging proposal into {} image entries".format(num_images))
    print(66*"=")

    agent_embedding = CLIP(model_size=args.embedding_model_size, device=args.device)
    e_vocabulary = agent_embedding.encode_text(full_vocabulary_names)

    # update annotations
    for this_img_entry in tqdm(gt_annos['images']):
        this_img_id = this_img_entry['id']

        spotlight_coi_names   = []
        # spotlight_coi_indices = []

        # not LLM-proposed yet
        if len(image_indexed_proposed_results[this_img_id]['vanilla_proposal_names']) <= .1:
            this_img_entry['spotlight_cois'] = [i + 1 for i in range(len(full_vocabulary_dict_with_synonyms))]
            continue

        cleaned_vanilla_proposal = remove_numbers(image_indexed_proposed_results[this_img_id]['vanilla_proposal_names'])

        e_query = agent_embedding.encode_text(cleaned_vanilla_proposal, truncate=True)

        # compute similarity scores
        mm_similarity = dot_score1(e_query, e_vocabulary)
        # Here, instead of finding the best buddy CoIs, find top-k similar CoIs
        top_k_scores, top_k_indices = torch.topk(mm_similarity, k=1, dim=1)

        # Collect all names that satisfies the requirements!
        for scores, indices in zip(top_k_scores, top_k_indices):
            for score, index in zip(scores, indices):
                if score > args.proposing_thresh:
                    spotlight_coi_names.append(full_vocabulary_names[index])

        # convert proposed CoI names to category_ids (1-indexed!)
        spotlight_coi_indices = [full_vocabulary_dict_with_synonyms[coi]['category_id'] for coi in spotlight_coi_names]
        spotlight_coi_indices = deduplicate_list(spotlight_coi_indices)
        # record it
        this_img_entry['spotlight_cois'] = sorted(spotlight_coi_indices)   # Miu: this is 1-indexed
        print(this_img_entry['spotlight_cois'])

    # <<< go over each image <<<
    destination_path = args.image_anno_path.replace(
        ".json",
        f"_spotdet_V2"
        f"_GPT35_inCLIP.json")

    dump_json(destination_path, gt_annos)
    print("Successfully saved merged results to {}".format(destination_path))

