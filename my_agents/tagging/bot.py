'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
 * modified by Mingxuan Liu
'''
import argparse
import numpy as np
import random
import os
import torch

from PIL import Image
from my_agents.tagging.ram.model import ram_plus
from my_agents.tagging.ram import inference_ram as inference
from my_agents.tagging.ram import inference_ram_openset as inference_openset
from my_agents.tagging.ram import get_transform

from my_agents.tagging.ram.utils import build_openset_llm_label_embedding
from torch import nn
import json


class RAMPP:
    def __init__(self, model_path, given_tags=None, threshold=0.5, image_size=384, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold,
        self.given_tags = given_tags        # json file
        self.image_size = 384
        self.model = None
        self.transform = get_transform(image_size)
        self.__create_model(threshold)

    def __create_model(self, threshold):
        self.model = ram_plus(pretrained=self.model_path, image_size=self.image_size, vit='swin_l')
        print("...Created RAM++ model from ckpt: {}".format(self.model_path))

        # Set open-set inference tags
        if self.given_tags is not None:
            openset_label_embedding, openset_categories = build_openset_llm_label_embedding(self.given_tags)
            self.model.tag_list = np.array(openset_categories)
            self.model.label_embed = nn.Parameter(openset_label_embedding.float())
            self.model.num_class = len(openset_categories)
            self.model.class_threshold = torch.ones(self.model.num_class) * threshold

        self.model.eval()
        self.model = self.model.to(self.device)

    def __tagging_4585(self, image):
        res = inference(image, self.model)
        return res[0]

    def __tagging_openset(self, image):
        res = inference_openset(image, self.model)
        return res

    def assign_tags(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)

        if self.given_tags is None:
            tags = self.__tagging_4585(image)
        else:
            tags = self.__tagging_openset(image)

        tags = tags.split("|")
        tags = [t.strip() for t in tags if len(t) >= 1]
        return tags