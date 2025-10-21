import logging
import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .registry_lvis_v1_val import register_lvis_instances


def get_lvis_22k_meta():
    from .lvis_22k_categories import CATEGORIES
    cat_ids = [k["id"] for k in CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


_CUSTOM_SPLITS_LVIS = {
    # Baseline
    "lvis_v1_val_baseline": ("coco/", "lvis/lvis_v1_val.json"),
}

_SPOTDET_SPLITS_LVIS_V2 = {
    # SpotDet
    # Ground Truth upper bound
    "lvis_v1_val_spotdet_v2_gt":                           ("coco/", "lvis/lvis_v1_val_spotdet_gt.json"),

    # # LLM
    # "lvis_v1_val_spotdet_v2_llm":                       ("coco/", "lvis/lvis_v1_val_spotdet_V2_GPT35.json"),
    # "lvis_v1_val_spotdet_v2_llm_inList":                ("coco/", "lvis/lvis_v1_val_spotdet_V2_GPT35_inList.json"),

    # CLIP
    "lvis_v1_val_spotdet_v2_clip_noun":                 ("coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=1_alpha=0.0_beta=0.0.json"),
    "lvis_v1_val_spotdet_v2_clip_noun_cap":             ("coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=1_alpha=0.5_beta=0.0.json"),
    "lvis_v1_val_spotdet_spotdet_v2_clip_noun_img":     ("coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=1_alpha=0.0_beta=0.5.json"),
    "lvis_v1_val_spotdet_spotdet_v2_clip_noun_cap_img": ("coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=1_alpha=0.3_beta=0.3.json"),

    # SBert
    "lvis_v1_val_spotdet_v2_sbert_noun":                ("coco/", "lvis/lvis_v1_val_spotdet_V2_sbert_synonyms=True_topk=1_alpha=0.0_beta=0.0.json"),
    "lvis_v1_val_spotdet_v2_sbert_noun_cap":            ("coco/", "lvis/lvis_v1_val_spotdet_V2_sbert_synonyms=True_topk=1_alpha=0.5_beta=0.0.json"),

    # K=3
    # CLIP
    "lvis_v1_val_spotdet_v2_clip_noun_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=3_alpha=0.0_beta=0.0.json"),
    "lvis_v1_val_spotdet_v2_clip_noun_cap_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=3_alpha=0.5_beta=0.0.json"),
    "lvis_v1_val_spotdet_spotdet_v2_clip_noun_img_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=3_alpha=0.0_beta=0.5.json"),
    "lvis_v1_val_spotdet_spotdet_v2_clip_noun_cap_img_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_clip_synonyms=True_topk=3_alpha=0.3_beta=0.3.json"),

    # SBert
    "lvis_v1_val_spotdet_v2_sbert_noun_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_sbert_synonyms=True_topk=3_alpha=0.0_beta=0.0.json"),
    "lvis_v1_val_spotdet_v2_sbert_noun_cap_k=3": (
    "coco/", "lvis/lvis_v1_val_spotdet_V2_sbert_synonyms=True_topk=3_alpha=0.5_beta=0.0.json"),

    # Tagging
    "lvis_v1_val_spotdet_v2_tagging_closeset": (
        "coco/", "lvis/lvis_v1_val_spotdet_V2_tagging_closeset_synonyms=True.json"),
    "lvis_v1_val_spotdet_v2_tagging_openset": (
        "coco/", "lvis/lvis_v1_val_spotdet_V2_tagging_openset_synonyms=True.json"),
}


def register_all_lvis_v1_val_only(root):
    # Standard LVIS
    for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS.items():
        register_lvis_instances(
            key,
            get_lvis_instances_meta("lvis_v1"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    # My experiments V2
    for key, (image_root, json_file) in _SPOTDET_SPLITS_LVIS_V2.items():
        register_lvis_instances(
            key,
            get_lvis_instances_meta("lvis_v1"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )