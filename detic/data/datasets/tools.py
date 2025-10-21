import os
from .registry_tools import register_tools_instances

"""
Dataset Notes
---
4 classes of surgical tools messed up by COCO 80 classes
"""

tool_categories = [
            {
            "id": 1,
            "name": "scalpel",
            "freebase_id": "tbd"
        },
        {
            "id": 2,
            "name": "straight dissection clamp",
            "freebase_id": "tbd"
        },
        {
            "id": 3,
            "name": "straight mayo scissor",
            "freebase_id": "tbd"
        },
        {
            "id": 4,
            "name": "curved mayo scissor",
            "freebase_id": "tbd"
        },
        {
            "id": 81,
            "name": "person",
            "freebase_id": "tbd"
        },
        {
            "id": 82,
            "name": "bicycle",
            "freebase_id": "tbd"
        },
        {
            "id": 83,
            "name": "car",
            "freebase_id": "tbd"
        },
        {
            "id": 84,
            "name": "motorcycle",
            "freebase_id": "tbd"
        },
        {
            "id": 5,
            "name": "airplane",
            "freebase_id": "tbd"
        },
        {
            "id": 6,
            "name": "bus",
            "freebase_id": "tbd"
        },
        {
            "id": 7,
            "name": "train",
            "freebase_id": "tbd"
        },
        {
            "id": 8,
            "name": "truck",
            "freebase_id": "tbd"
        },
        {
            "id": 9,
            "name": "boat",
            "freebase_id": "tbd"
        },
        {
            "id": 10,
            "name": "traffic light",
            "freebase_id": "tbd"
        },
        {
            "id": 11,
            "name": "fire hydrant",
            "freebase_id": "tbd"
        },
        {
            "id": 12,
            "name": "stop sign",
            "freebase_id": "tbd"
        },
        {
            "id": 13,
            "name": "parking meter",
            "freebase_id": "tbd"
        },
        {
            "id": 14,
            "name": "bench",
            "freebase_id": "tbd"
        },
        {
            "id": 15,
            "name": "bird",
            "freebase_id": "tbd"
        },
        {
            "id": 16,
            "name": "cat",
            "freebase_id": "tbd"
        },
        {
            "id": 17,
            "name": "dog",
            "freebase_id": "tbd"
        },
        {
            "id": 18,
            "name": "horse",
            "freebase_id": "tbd"
        },
        {
            "id": 19,
            "name": "sheep",
            "freebase_id": "tbd"
        },
        {
            "id": 20,
            "name": "cow",
            "freebase_id": "tbd"
        },
        {
            "id": 21,
            "name": "elephant",
            "freebase_id": "tbd"
        },
        {
            "id": 22,
            "name": "bear",
            "freebase_id": "tbd"
        },
        {
            "id": 23,
            "name": "zebra",
            "freebase_id": "tbd"
        },
        {
            "id": 24,
            "name": "giraffe",
            "freebase_id": "tbd"
        },
        {
            "id": 25,
            "name": "backpack",
            "freebase_id": "tbd"
        },
        {
            "id": 26,
            "name": "umbrella",
            "freebase_id": "tbd"
        },
        {
            "id": 27,
            "name": "handbag",
            "freebase_id": "tbd"
        },
        {
            "id": 28,
            "name": "tie",
            "freebase_id": "tbd"
        },
        {
            "id": 29,
            "name": "suitcase",
            "freebase_id": "tbd"
        },
        {
            "id": 30,
            "name": "frisbee",
            "freebase_id": "tbd"
        },
        {
            "id": 31,
            "name": "skis",
            "freebase_id": "tbd"
        },
        {
            "id": 32,
            "name": "snowboard",
            "freebase_id": "tbd"
        },
        {
            "id": 33,
            "name": "sports ball",
            "freebase_id": "tbd"
        },
        {
            "id": 34,
            "name": "kite",
            "freebase_id": "tbd"
        },
        {
            "id": 35,
            "name": "baseball bat",
            "freebase_id": "tbd"
        },
        {
            "id": 36,
            "name": "baseball glove",
            "freebase_id": "tbd"
        },
        {
            "id": 37,
            "name": "skateboard",
            "freebase_id": "tbd"
        },
        {
            "id": 38,
            "name": "surfboard",
            "freebase_id": "tbd"
        },
        {
            "id": 39,
            "name": "tennis racket",
            "freebase_id": "tbd"
        },
        {
            "id": 40,
            "name": "bottle",
            "freebase_id": "tbd"
        },
        {
            "id": 41,
            "name": "wine glass",
            "freebase_id": "tbd"
        },
        {
            "id": 42,
            "name": "cup",
            "freebase_id": "tbd"
        },
        {
            "id": 43,
            "name": "fork",
            "freebase_id": "tbd"
        },
        {
            "id": 44,
            "name": "knife",
            "freebase_id": "tbd"
        },
        {
            "id": 45,
            "name": "spoon",
            "freebase_id": "tbd"
        },
        {
            "id": 46,
            "name": "bowl",
            "freebase_id": "tbd"
        },
        {
            "id": 47,
            "name": "banana",
            "freebase_id": "tbd"
        },
        {
            "id": 48,
            "name": "apple",
            "freebase_id": "tbd"
        },
        {
            "id": 49,
            "name": "sandwich",
            "freebase_id": "tbd"
        },
        {
            "id": 50,
            "name": "orange",
            "freebase_id": "tbd"
        },
        {
            "id": 51,
            "name": "broccoli",
            "freebase_id": "tbd"
        },
        {
            "id": 52,
            "name": "carrot",
            "freebase_id": "tbd"
        },
        {
            "id": 53,
            "name": "hot dog",
            "freebase_id": "tbd"
        },
        {
            "id": 54,
            "name": "pizza",
            "freebase_id": "tbd"
        },
        {
            "id": 55,
            "name": "donut",
            "freebase_id": "tbd"
        },
        {
            "id": 56,
            "name": "cake",
            "freebase_id": "tbd"
        },
        {
            "id": 57,
            "name": "chair",
            "freebase_id": "tbd"
        },
        {
            "id": 58,
            "name": "couch",
            "freebase_id": "tbd"
        },
        {
            "id": 59,
            "name": "potted plant",
            "freebase_id": "tbd"
        },
        {
            "id": 60,
            "name": "bed",
            "freebase_id": "tbd"
        },
        {
            "id": 61,
            "name": "dining table",
            "freebase_id": "tbd"
        },
        {
            "id": 62,
            "name": "toilet",
            "freebase_id": "tbd"
        },
        {
            "id": 63,
            "name": "tv",
            "freebase_id": "tbd"
        },
        {
            "id": 64,
            "name": "laptop",
            "freebase_id": "tbd"
        },
        {
            "id": 65,
            "name": "mouse",
            "freebase_id": "tbd"
        },
        {
            "id": 66,
            "name": "remote",
            "freebase_id": "tbd"
        },
        {
            "id": 67,
            "name": "keyboard",
            "freebase_id": "tbd"
        },
        {
            "id": 68,
            "name": "cell phone",
            "freebase_id": "tbd"
        },
        {
            "id": 69,
            "name": "microwave",
            "freebase_id": "tbd"
        },
        {
            "id": 70,
            "name": "oven",
            "freebase_id": "tbd"
        },
        {
            "id": 71,
            "name": "toaster",
            "freebase_id": "tbd"
        },
        {
            "id": 72,
            "name": "sink",
            "freebase_id": "tbd"
        },
        {
            "id": 73,
            "name": "refrigerator",
            "freebase_id": "tbd"
        },
        {
            "id": 74,
            "name": "book",
            "freebase_id": "tbd"
        },
        {
            "id": 75,
            "name": "clock",
            "freebase_id": "tbd"
        },
        {
            "id": 76,
            "name": "vase",
            "freebase_id": "tbd"
        },
        {
            "id": 77,
            "name": "scissors",
            "freebase_id": "tbd"
        },
        {
            "id": 78,
            "name": "teddy bear",
            "freebase_id": "tbd"
        },
        {
            "id": 79,
            "name": "hair drier",
            "freebase_id": "tbd"
        },
        {
            "id": 80,
            "name": "toothbrush",
            "freebase_id": "tbd"
        }
]

# register the hierarchical datasets
_PREDEFINED_TOOLS_CATEGORIES = {
    "tools_val": tool_categories,  # num_classes = 84
    "tools_val_gt": tool_categories,  # num_classes = 84

}

_PREDEFINED_SPLITS_TOOLS = {
    # |- test set
    "tools_val": ("labeled_tools/images", "labeled_tools/annotations/annotation_tools2024.json"),
    "tools_val_gt": ("labeled_tools/images", "labeled_tools/annotations/annotation_tools2024_GT.json"),

    # "fsod_test_l2": ("fsod/images/", "fsod/annotations/fsod_test_fixed_l2.json"),
    # "fsod_test_l3": ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3.json"),
    # # |- toy test set
    # "fsod_test_toy200": ("fsod/images_toy/", "fsod/annotations/fsod_test_toy200.json"),
    # # SpotDet
    # # upper-bound
    # "fsod_test_l3_spotdet_gt":                ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_gt.json"),
    # # LLM
    # "fsod_test_l3_spotdet_llm":               ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_GPT35.json"),
    # # CLIP
    # "fsod_test_l3_spotdet_clip_noun":         ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_clip_topk=3_alpha=0.0_beta=0.0.json"),
    # "fsod_test_l3_spotdet_clip_noun_cap":     ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_clip_topk=3_alpha=0.5_beta=0.0.json"),
    # "fsod_test_l3_spotdet_clip_noun_img":     ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_clip_topk=3_alpha=0.0_beta=0.5.json"),
    # "fsod_test_l3_spotdet_clip_noun_cap_img": ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_clip_topk=3_alpha=0.3_beta=0.3.json"),
    # # SBert
    # "fsod_test_l3_spotdet_sbert_noun":        ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_sbert_topk=3_alpha=0.0_beta=0.0.json"),
    # "fsod_test_l3_spotdet_sbert_noun_cap":    ("fsod/images/", "fsod/annotations/fsod_test_fixed_l3_spotdet_sbert_topk=3_alpha=0.5_beta=0.0.json"),
}


def _get_builtin_metadata(cats):
    id_to_name = {x['id']: x['name'] for x in cats}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(cats))}    # convert to 1-indexed
    thing_classes = [x['name'] for x in sorted(cats, key=lambda x: x['id'])]    # sorted name list that match the 1-indexed list
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


def register_all_tools(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TOOLS.items():
        register_tools_instances(
            name=key,
            metadata=_get_builtin_metadata(_PREDEFINED_TOOLS_CATEGORIES[key]),
            json_file=os.path.join(root, json_file) if "://" not in json_file else json_file,
            image_root=os.path.join(root, image_root),  # dataset root
        )