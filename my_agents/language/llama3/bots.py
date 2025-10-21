# Written by Mingxuan Liu

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# context-length: 8k
_MODEL_ZOO = {
    # 8B models take ~15GB GPU-mem to load
    'llama3-8b': "meta-llama/Meta-Llama-3-8B",
    'llama3-8b-instruct': "meta-llama/Meta-Llama-3-8B-Instruct",
    # 70B models take ~150GB GPU-mem to load
    'llama3-70b': "meta-llama/Meta-Llama-3-70B",
    'llama3-70b-instruct': "meta-llama/Meta-Llama-3-70B-Instruct",
}


class LLaMA3sAlpha:
    def __init__(
            self,
            model='llama3-8b-instruct',
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=256,
            do_sample=True
    ):
        self.model_name = model
        self.model_id = _MODEL_ZOO[model]
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        # llama model pipeline
        self.pipeline = None
        self.terminators = None
        # build pipeline
        self.__initialize()

    def __initialize(
            self
    ):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def reset_temperature(
            self,
            temperature
    ):
        self.temperature = temperature

    def reset_top_p(
            self,
            top_p
    ):
        self.top_p = top_p

    def reset_do_sample(
            self,
            do_sample
    ):
        self.do_sample = do_sample

    def reset_max_new_tokens(
            self,
            max_new_tokens
    ):
        self.max_new_tokens = max_new_tokens

    def get_used_tokens(
            self
    ):
        # dummy function
        return self.max_new_tokens

    def __prepare_messages(
            self,
            text,
            system_prompt=None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": f"{system_prompt}"}
            )
        else:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )

        messages.append(
            {"role": "user", "content": f"{text}"}
        )
        return messages

    def __prepare_prompt(
            self,
            messages
    ):
        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def __query(
            self,
            prompt,
    ):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs[0]["generated_text"][len(prompt):].strip()


    def infer(
            self,
            text,
            system_prompt=None,
    ):
        # print(text)
        messages = self.__prepare_messages(text, system_prompt)
        prompt = self.__prepare_prompt(messages)
        replied_text = self.__query(prompt)
        return replied_text


class LLaMA3sBeta:
    def __init__(
            self,
            model='llama3-8b-instruct',
            pretrained_model_path=None,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=256,
            do_sample=True
    ):
        self.model_name = model
        self.model_id = _MODEL_ZOO[model]
        self.pre_trained_model_path = pretrained_model_path

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        # llama model pipeline
        self.tokenizer = None
        self.model = None
        self.terminators = None
        # build models
        self.__initialize()

    def __initialize(
            self
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def reset_temperature(
            self, temperature
    ):
        self.temperature = temperature

    def reset_top_p(
            self,
            top_p
    ):
        self.top_p = top_p

    def reset_do_sample(
            self,
            do_sample
    ):
        self.do_sample = do_sample

    def reset_max_new_tokens(
            self,
            max_new_tokens
    ):
        self.max_new_tokens = max_new_tokens

    def __prepare_messages(
            self,
            text,
            system_prompt=None,
    ):
        messages = []
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": f"{system_prompt}"}
            )
        else:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )

        messages.append(
            {"role": "user", "content": f"{text}"}
        )
        return messages

    def __prepare_prompt(
            self,
            messages
    ):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        return prompt

    def __query(
            self,
            prompt
    ):
        outputs = self.model.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return outputs[0][prompt.shape[-1]:]

    def __decode(
            self,
            outputs
    ):
        return self.tokenizer.decode(outputs, skip_special_tokens=True)

    def infer(
            self,
            text,
            system_prompt=None,
    ):
        messages = self.__prepare_messages(text, system_prompt)
        prompt = self.__prepare_prompt(messages)
        outputs = self.__query(prompt)
        replied_text = self.__decode(outputs)
        return replied_text



def debug_test():
    text = "In terms of basketball playing, Kobe and James who is better? What are their roles in NBA?"

    llm_alpha = LLaMA3sAlpha(model='llama3-8b-instruct')
    reply_alpha = llm_alpha.infer(text)
    print(reply_alpha)

    print("\n\n")


    llm_beta = LLaMA3sBeta(model='llama3-8b-instruct')
    reply_beta = llm_beta.infer(text)
    print(reply_beta)


if __name__ == "__main__":
    debug_test()

