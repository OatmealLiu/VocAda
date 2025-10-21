# Written by Mingxuan Liu
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import matplotlib.pyplot as plt

from my_agents.fileios import *

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

from PIL import Image
import math

import spacy


TOP_MISS_LLM = [
    "cup", "tv", "handbag", "person", "chair",
    "bottle", "car", "truck", "book", "bowl",
]


def parse_performance(text):
    l_text = text.split('|')
    l_text = [e.strip() for e in l_text if e.strip() != '']
    rlt = {}
    l1 = list(range(0, len(l_text), 2))
    l2 = list(range(1, len(l_text), 2))
    for i, j in zip(l1, l2):
        rlt[l_text[i]] = l_text[j]
    return rlt


def get_diff(a, b):
    diff = {}
    for k in a.keys():
        if a[k] == 'nan':
            continue
        diff[k] = float(b[k]) - float(a[k])
    return diff


def plot_performance_diff(path, miss_class_counter, title="Baseline v.s. GT", height=20):
    # Sort dictionary by its values (miss counts) and unpack into lists for plotting
    items = sorted(miss_class_counter.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)

    # Plotting
    plt.figure(figsize=(10, height))
    plt.barh(labels, values)  # Displaying top 20 for better visualization
    plt.xlabel('Performance difference')
    plt.ylabel('Class Names')
    plt.title(title)
    plt.tight_layout()
    # Save the plot as a PDF file
    plt.savefig(path)


def plot_miss_gts_statistics(path, miss_class_counter, title="Viz"):
    # Sort dictionary by its values (miss counts) and unpack into lists for plotting
    items = sorted(miss_class_counter.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)

    # Plotting
    plt.figure(figsize=(10, 14))
    plt.barh(labels, values)  # Displaying top 20 for better visualization
    plt.xlabel('Proposal Miss Count')
    plt.ylabel('Class Names')
    plt.title(title)
    plt.tight_layout()
    # Save the plot as a PDF file
    plt.savefig(path)


def eval_proposal(args):
    """
    True  Positive (TP): correctly proposed CoIs
    False Positive (FP): wrongly proposed CoIs
    False Negative (FN): miss GTs

    Metrics:
    gt_hit_rate : hit_gt / total_gts
    precision   : # TP / TP + FP
    recall      : # TP / TP + FN
    """
    gt_proposal      = load_json(args.image_anno_path.replace(".json", "_spotdet_gt.json"))
    spotdet_proposal = load_json(args.image_anno_path.replace(".json", f"_spotdet_GPT35.json")) \
        if args.pipeline_proposing == 'llm' else load_json(args.image_anno_path.replace(".json", f"_spotdet_{args.embedding_model_name}_topk={args.proposing_topk}_alpha={args.proposing_alpha}_beta={args.proposing_beta}.json"))

    full_vocabulary_dict  = class_names_v2[args.dataset_name]
    baseline_proposals = list(full_vocabulary_dict.values())
    mapper_id2name = {v: k for k, v in full_vocabulary_dict.items()}

    gt_proposed_cois      = {entry['id']: entry['spotlight_cois'] for entry in gt_proposal['images']}
    spotdet_proposed_cois = {entry['id']: entry['spotlight_cois'] for entry in spotdet_proposal['images']}

    num_total_miss    =  0
    miss_counter      = {k: 0 for k in full_vocabulary_dict.keys()}
    miss_file_names   = {k: [] for k in full_vocabulary_dict.keys()}
    gt_hit_rate       = .0          # hit_gt / total_gts
    precision         = .0          # TP / TP + FP
    recall            = .0          # TP / TP + FN
    jaccard_index     = .0
    for i, (image_id, gt_cois) in tqdm(enumerate(gt_proposed_cois.items())):
        pred_cois = list(set(spotdet_proposed_cois[image_id])) if args.pipeline_proposing != "baseline" \
            else baseline_proposals

        print(f"Gt = {gt_cois}    |    Pd = {pred_cois}")

        tp = .0
        fp = .0
        fn = .0
        for gt_coi in gt_cois:
            if gt_coi in pred_cois:
                tp += 1.0
            else:
                fn += 1.0
                num_total_miss += 1
                miss_counter[mapper_id2name[gt_coi]] += 1
                miss_file_names[mapper_id2name[gt_coi]].append(image_id)

        for pred_coi in pred_cois:
            if pred_coi not in gt_cois:
                fp += 1.0

        gt_hit_rate   += (tp / len(gt_cois))
        precision     += (tp / (tp + fp))
        recall        += (tp / (tp + fn))
        jaccard_index += (tp / len(gt_cois) + len(pred_cois))

    sorted_miss_counter = dict(sorted(miss_counter.items(), key=lambda item: item[1]))
    sorted_miss_file_names = {k: miss_file_names[k] for k in sorted_miss_counter.keys()}
    gt_hit_rate   /= len(gt_proposed_cois)
    precision     /= len(gt_proposed_cois)
    recall        /= len(gt_proposed_cois)
    jaccard_index /= len(gt_proposed_cois)

    results = {
        "num_total_miss":         num_total_miss,
        "gt_hit_rate":            gt_hit_rate,
        "precision":              precision,
        "recall":                 recall,
        "jaccard_index":          jaccard_index,
        "miss_class_counter":     sorted_miss_counter,
        "sorted_miss_file_names": sorted_miss_file_names
    }

    if args.pipeline_proposing == 'baseline':
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}_spotdet_baseline(full_vocabulary).json")
    elif args.pipeline_proposing == 'llm':
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}_spotdet_GPT35.json")
    else:
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}_spotdet_{args.embedding_model_name}_topk={args.proposing_topk}_alpha={args.proposing_alpha}_beta={args.proposing_beta}.json")

    plot_miss_gts_statistics(destination_path.replace(".json", ".pdf"),
                             results["miss_class_counter"],
                             title=args.pipeline_proposing)

    dump_json(destination_path, results)
    print("Successfully saved merged results to {}".format(destination_path))


def eval_proposal_v3(args):
    """
    True  Positive (TP): correctly proposed CoIs
    False Positive (FP): wrongly proposed CoIs
    False Negative (FN): miss GTs

    Metrics:
    gt_hit_rate : hit_gt / total_gts
    precision   : # TP / TP + FP
    recall      : # TP / TP + FN
    """
    gt_proposal      = load_json(args.image_anno_path.replace(".json", "_spotdet_gt.json"))

    if args.pipeline_proposing == 'llm':
        spotdet_proposal = load_json(
            args.image_anno_path.replace(
                ".json",
                f"_spotdet_V2"
                f"_GPT35_inList.json")
        )
    elif args.pipeline_proposing == 'tagging':
        # Construct the new file name with replacements and formatting
        spotdet_proposal = load_json(
            args.image_anno_path.replace(
                ".json",
                f"_spotdet_V2"
                f"_tagging"
                f"_openset"
                f"_synonyms={args.use_synonyms}"
                f".json")
        )
    else:
        spotdet_proposal = load_json(
            args.image_anno_path.replace(
                ".json",
                f"_spotdet_V2"
                f"_{args.embedding_model_name}"
                f"_synonyms={args.use_synonyms}"
                f"_topk={args.proposing_topk}"
                f"_alpha={args.proposing_alpha}_beta={args.proposing_beta}"
                f".json")
        )

    full_vocabulary_dict  = class_names_v3[args.dataset_name]
    baseline_proposals    = [v['category_id'] for v in full_vocabulary_dict.values()]
    mapper_id2name        = {v['category_id']: k for k, v in full_vocabulary_dict.items()}

    gt_proposed_cois      = {entry['id']: entry['spotlight_cois'] for entry in gt_proposal['images']}
    spotdet_proposed_cois = {entry['id']: entry['spotlight_cois'] for entry in spotdet_proposal['images']}

    num_total_miss    = 0
    miss_counter      = {k: 0 for k in full_vocabulary_dict.keys()}
    miss_file_names   = {k: [] for k in full_vocabulary_dict.keys()}
    gt_hit_rate       = .0          # hit_gt / total_gts
    precision         = .0          # TP / TP + FP
    recall            = .0          # TP / TP + FN
    jaccard_index     = .0
    for i, (image_id, gt_cois) in tqdm(enumerate(gt_proposed_cois.items())):
        pred_cois = list(set(spotdet_proposed_cois[image_id])) if args.pipeline_proposing != "baseline" \
            else baseline_proposals

        print(f"Gt = {gt_cois}    |    Pd = {pred_cois}")

        tp = .0
        fp = .0
        fn = .0
        for gt_coi in gt_cois:
            if gt_coi in pred_cois:
                tp += 1.0
            else:
                fn += 1.0
                num_total_miss += 1
                miss_counter[mapper_id2name[gt_coi]] += 1
                miss_file_names[mapper_id2name[gt_coi]].append(image_id)

        for pred_coi in pred_cois:
            if pred_coi not in gt_cois:
                fp += 1.0

        gt_hit_rate   += (tp / len(gt_cois)) if len(gt_cois) > 0 else 0
        precision     += (tp / (tp + fp))
        recall        += (tp / (tp + fn)) if (tp + fn) > 0 else 0
        jaccard_index += (tp / len(gt_cois) + len(pred_cois)) if len(gt_cois) > 0 else 0

    sorted_miss_counter = dict(sorted(miss_counter.items(), key=lambda item: item[1]))
    sorted_miss_file_names = {k: miss_file_names[k] for k in sorted_miss_counter.keys()}
    gt_hit_rate   /= len(gt_proposed_cois)
    precision     /= len(gt_proposed_cois)
    recall        /= len(gt_proposed_cois)
    jaccard_index /= len(gt_proposed_cois)

    results = {
        "num_total_miss":         num_total_miss,
        "gt_hit_rate":            gt_hit_rate,
        "precision":              precision,
        "recall":                 recall,
        "jaccard_index":          jaccard_index,
        "miss_class_counter":     sorted_miss_counter,
        "sorted_miss_file_names": sorted_miss_file_names
    }

    if args.pipeline_proposing == 'baseline':
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}"
                                                             f"_spotdet_V2"
                                                             f"_baseline(full_vocabulary).json")
    elif args.pipeline_proposing == 'llm':
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}"
                                                             f"_spotdet_V2"
                                                             f"_GPT35_inList.json")
    elif args.pipeline_proposing == 'tagging':
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}"
                                                             f"_spotdet_V2"
                                                             f"_tagging_openset.json")
    else:
        destination_path = os.path.join(args.answers_folder, f"eval_proposal_{args.dataset_name}"
                                                             f"_spotdet_V2"
                                                             f"_{args.embedding_model_name}"
                                                             f"_synonyms={args.use_synonyms}"
                                                             f"_topk={args.proposing_topk}"
                                                             f"_alpha={args.proposing_alpha}_beta={args.proposing_beta}"
                                                             f".json")

    plot_miss_gts_statistics(destination_path.replace(".json", ".pdf"),
                             results["miss_class_counter"],
                             title=args.pipeline_proposing)

    dump_json(destination_path, results)
    print("Successfully saved merged results to {}".format(destination_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str,
                        default="coco8",
                        choices=["coco8", "coco", "fsod_toy200",
                                 "lvis", "oid", "object365", "v3det",
                                 "coco_full80", "fsod_full200"])
    parser.add_argument("--query-mode", type=str,
                        default="captioning",
                        choices=["captioning", "merging", "proposing", "merging_llm_proposals"])
    parser.add_argument("--pipeline-proposing", type=str,
                        default='embedding',
                        choices=["llm", "embedding", "ground_truth", 'baseline', 'tagging'])
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
    parser.add_argument('--use-synonyms', default=True)

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

    # do evaluation
    # eval_proposal(args)
    eval_proposal_v3(args)
