# Written by Mingxuan Liu

import os
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


_API_BASE = {
    "gpt-3.5-turbo": "https://api.openai.com/v1",           # context window: 16k
    "gpt-3.5-turbo-0125": "https://api.openai.com/v1",      # context window: 16k
    "gpt-4": "https://api.openai.com/v1",                   # context window: 8k
    "gpt-4-turbo": "https://api.openai.com/v1",  # context window: 128k
    "gpt-4-turbo-preview": "https://api.openai.com/v1",     # context window: 128k
    "gpt-4-0125-preview": "https://api.openai.com/v1",      # context window: 128k
}


SYSTEM_INSTRUCTION = "You are a helpful assistant."


def prepare_chatgpt_message(
        main_prompt,
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
        {"role": "user", "content": f"{main_prompt}"}
    )
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpts(client, chatgpt_messages, temperature=0.99, max_tokens=40, model="gpt-4-turbo-preview"):
    if max_tokens > 0:
        response = client.chat.completions.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=chatgpt_messages,
            temperature=temperature
        )

    reply = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    return reply, total_tokens


def trim_question(question):
    question = question.split('Question: ')[-1].replace('\n', ' ').strip()
    if 'Answer:' in question:  # Some models make up an answer after asking. remove it
        q, a = question.split('Answer:')[:2]
        if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
            question = a.strip()
        else:
            question = q.strip()
    return question


class GPTS:
    def __init__(
            self,
            model="gpt-4-turbo",
            temperature=0.99,
            max_chat_token=-1,
    ):
        if model not in _API_BASE:
            raise ValueError

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            # base_url=_API_BASE[model],
        )
        self.model_name = model
        self.max_chat_token = max_chat_token
        self.temperature = temperature
        self.total_tokens = 0

    def reset(
            self
    ):
        self.total_tokens = 0

    def get_used_tokens(
            self
    ):
        return self.total_tokens

    def get_model_name(
            self
    ):
        return self.model_name

    def __query(
            self,
            prompt,
            system_prompt=None,
    ):
        total_prompt = prepare_chatgpt_message(prompt, system_prompt)
        reply, n_tokens = call_gpts(
            client=self.client,
            chatgpt_messages=total_prompt,
            temperature=self.temperature,
            model=self.model_name,
            max_tokens=self.max_chat_token,
            )

        return reply, total_prompt, n_tokens

    def infer(
            self,
            prompt,
            system_prompt=None,
    ):
        reply, _, n_tokens = self.__query(prompt, system_prompt)
        self.total_tokens += n_tokens
        return reply.strip()


def test_get_llm_output():
    prompt = "hello"

    model = "gpt-4-turbo-preview"
    bot_llm = GPTS(model=model)
    completion = bot_llm.infer(prompt)
    token = bot_llm.get_used_tokens()
    print(f"{model=}, {completion=}, {token=}")

    model = "gpt-3.5-turbo"
    bot_llm = GPTS(model=model)
    completion = bot_llm.infer(prompt)
    token = bot_llm.get_used_tokens()
    print(f"{model=}, {completion=}, {token=}")


if __name__ == '__main__':
    test_get_llm_output()
