# Written by Mingxuan Liu

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


BLIP2ZOO = {
    'FlanT5-XXL': 'Salesforce/blip2-flan-t5-xxl',
    'FlanT5-XL-COCO': 'Salesforce/blip2-flan-t5-xl-coco',
    'FlanT5-XL': 'Salesforce/blip2-flan-t5-xl',
    'OPT6.7B-COCO': 'Salesforce/blip2-opt-6.7b-coco',
    'OPT2.7B-COCO': 'Salesforce/blip2-opt-2.7b-coco',
    'OPT6.7B': 'Salesforce/blip2-opt-6.7b',
    'OPT2.7B': 'Salesforce/blip2-opt-2.7b',
}

ANSWER_INSTRUCTION = 'Answer given questions. If you are not sure about the answer, say you don\'t ' \
                     'know honestly. Don\'t imagine any contents that are not in the image.'

SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following blip2 huggingface demo


def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]  # remove the last '/n'
    return chat_log


def trim_answer(answer):
    answer = answer.split('Question:')[0].replace('\n', ' ').strip()
    return answer


class BLIP2:
    def __init__(self, model_tag, device='cpu', device_id=0, bit8=False, max_answer_tokens=-1):
        # load BLIP-2 to a single gpu
        self.model_tag = model_tag
        self.model_name = "BLIP-2 " + self.model_tag
        self.max_answer_tokens = max_answer_tokens
        self.model_dtype = torch.bfloat16 if 'FlanT5' in self.model_tag else torch.float16

        self.blip2_processor = Blip2Processor.from_pretrained(BLIP2ZOO[self.model_tag])
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag],
                                                                   # device_map={'': int(device_id)},
                                                                   # **dtype,
                                                                   torch_dtype=self.model_dtype,
                                                                   device_map='auto',
                                                                   )
        if device == 'cpu':
            self.device = 'cpu'
            self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag])
        else:
            self.device = 'cuda:{}'.format(device_id)
            self.bit8 = bit8
            # dtype = {'load_in_8bit': True} if self.bit8 else {'torch_dtype': torch.float16}
            dtype = {}
            if self.bit8:
                dtype['load_in_8bit'] = True
            else:
                dtype['torch_dtype'] = self.model_dtype

            self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag],
                                                                       device_map={'': int(device_id)},
                                                                       **dtype,
                                                                       )

    def __call_blip2(self, raw_image, prompt):
        if self.device == 'cpu':
            inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt")
        else:
            inputs = self.blip2_processor(raw_image, prompt, return_tensors="pt").to(self.device,
                                                                                     self.model_dtype)

        out = self.blip2.generate(**inputs,  max_new_tokens=self.max_answer_tokens) \
            if self.max_answer_tokens > 0 else self.blip2.generate(**inputs)

        reply = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        # reply = self.blip2_processor.batch_decode(out, skip_special_tokens=True)[0].strip()   # official
        return reply

    def get_model_name(self):
        return self.model_name

    def do_vqa(self, raw_image, prompt):
        # prompt = f"Questions: {text}"
        reply = self.__call_blip2(raw_image, prompt)
        return reply.strip()

    def caption(self, raw_image):
        # starndard way to caption an image in the blip2 paper
        std_prompt = 'a photo of'
        reply = self.__call_blip2(raw_image, std_prompt)
        reply = reply.replace('\n', ' ').strip()  # trim caption
        return reply.strip()

    def answer_chat_log(self, raw_image, chat_log, n_blip2_context=-1):
        # prepare the context for blip2
        blip2_prompt = '\n'.join([ANSWER_INSTRUCTION,
                                  get_chat_log(chat_log['questions'],chat_log['answers'],
                                               last_n=n_blip2_context), SUB_ANSWER_INSTRUCTION]
                                 )

        reply = self.__call_blip2(raw_image, blip2_prompt)
        return reply.strip()

    def call_llm(self, prompts):
        prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp['input_ids'].to(self.device)
        attention_mask = prompts_temp['attention_mask'].to(self.device, torch.float16)

        prompts_embeds = self.blip2.language_model.get_input_embeddings()(input_ids)

        outputs = self.blip2.language_model.generate(
            inputs_embeds=prompts_embeds,
            attention_mask=attention_mask)

        outputs = self.blip2_processor.decode(outputs[0], skip_special_tokens=True)
        return outputs.strip()


class BLIP2Beta:
    def __init__(
            self,
            model_tag,
            device_id=0,
            max_answer_tokens=256
    ):
        # load BLIP-2 to a single gpu
        self.model_tag = model_tag
        self.model_name = "BLIP-2 " + self.model_tag
        self.max_answer_tokens = max_answer_tokens

        self.model_dtype = torch.bfloat16 if 'FlanT5' in self.model_tag else torch.float16

        self.processor = Blip2Processor.from_pretrained(BLIP2ZOO[self.model_tag])
        self.model     = Blip2ForConditionalGeneration.from_pretrained(BLIP2ZOO[self.model_tag],
                                                                       torch_dtype=self.model_dtype,
                                                                       # device_map='auto',
                                                                       device_map={'': int(device_id)},
                                                                       )


    def __prepare_inputs(self, raw_image, prompt):
        inputs = self.processor(raw_image, prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def __query(self, inputs):
        outputs = self.model.generate(**inputs,  max_new_tokens=self.max_answer_tokens)
        # outputs = self.model.generate(**inputs)

        # reply = self.processor.decode(outputs[0], skip_special_tokens=True)
        reply = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]   # official
        return reply.strip()

    def get_model_name(self):
        return self.model_name

    def do_vqa(self, raw_image, prompt):
        inputs = self.__prepare_inputs(raw_image, prompt)
        replied_text = self.__query(inputs)
        return replied_text

    # def caption(self, raw_image):
    #     # starndard way to caption an image in the blip2 paper
    #     std_prompt = 'a photo of'
    #     reply = self.__call_blip2(raw_image, std_prompt)
    #     reply = reply.replace('\n', ' ').strip()  # trim caption
    #     return reply.strip()

    # def answer_chat_log(self, raw_image, chat_log, n_blip2_context=-1):
    #     # prepare the context for blip2
    #     blip2_prompt = '\n'.join([ANSWER_INSTRUCTION,
    #                               get_chat_log(chat_log['questions'],chat_log['answers'],
    #                                            last_n=n_blip2_context), SUB_ANSWER_INSTRUCTION]
    #                              )
    #
    #     reply = self.__call_blip2(raw_image, blip2_prompt)
    #     return reply.strip()

    # def call_llm(self, prompts):
    #     prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
    #     input_ids = prompts_temp['input_ids'].to(self.device)
    #     attention_mask = prompts_temp['attention_mask'].to(self.device, torch.float16)
    #
    #     prompts_embeds = self.blip2.language_model.get_input_embeddings()(input_ids)
    #
    #     outputs = self.blip2.language_model.generate(
    #         inputs_embeds=prompts_embeds,
    #         attention_mask=attention_mask)
    #
    #     outputs = self.blip2_processor.decode(outputs[0], skip_special_tokens=True)
    #     return outputs.strip()