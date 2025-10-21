# Written by Mingxuan Liu

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from my_agents.vision.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from my_agents.vision.llava.conversation import conv_templates, SeparatorStyle
from my_agents.vision.llava.model.builder import load_pretrained_model
from my_agents.vision.llava.utils import disable_torch_init
from my_agents.vision.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from my_agents.fileios import *
from PIL import Image
import math


_LLAVA_MODEL_ZOO = {
    'llava-v1.6-mistral-7b': {
        'llava_size': '7b',
        'llava_base': 'mistral',
        'llava_version': 'v1.6',
        'llava_conv_mode': ['mistral_instruct', 'llava_v1'],
    },
    'llava-v1.6-34b': {
        'llava_size': '34b',
        'llava_base': 'nous-hermes-2-yi',
        'llava_version': 'v1.6',
        'llava_conv_mode': ['chatml_direct', 'llava_v1'],
    },
}


class LLaVA:
    def __init__(self, model_path, model_base, conv_mode, conv_adapt=False, temperature=0.2, top_p=None, num_beams=1):
        # config
        self.model_path = os.path.expanduser(model_path)
        self.model_name = get_model_name_from_path(model_path)
        self.model_base = model_base
        self.conv_mode = _LLAVA_MODEL_ZOO[self.model_name]["llava_conv_mode"][0] if conv_adapt else conv_mode
        # hyerparameters
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.__create_llava()

    def __create_llava(self):
        # disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, self.model_name)

    def __call_llava(self, raw_image, text):
        image_sizes = raw_image.size
        image_tensor = process_images([raw_image], self.image_processor, self.model.config)[0]

        # Curate the question text a bit following LLaVA way
        cur_prompt = text
        if self.model.config.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        # Initiate the conversation
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()  # get the input txt prompt

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()

        # Actual LLaVA query inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image_sizes],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )
        # Parse the output results
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        results = {
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": self.model_name,
            "conv_mode": self.conv_mode,
            "metadata": {},
        }
        return results

    def get_model_name(self):
        return self.model_name

    def do_vqa(self, raw_image, text):
        # std_prompt = f"Questions: {text}"
        reply = self.__call_llava(raw_image, text)
        return reply

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        std_prompt = 'Describe this photo in detail'
        reply = self.__call_llava(raw_image, std_prompt)
        return reply

