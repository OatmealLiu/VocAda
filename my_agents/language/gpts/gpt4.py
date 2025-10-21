import json
import logging
import os
import threading
from typing import List
from openai import OpenAI

API_BASE = {
    "gpt-3.5-turbo": "https://api.openai.com/v1",
    "gpt-4": "https://api.openai.com/v1",
}


def get_llm_output(prompt: str, model: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=API_BASE[model])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            response = completion.choices[0].message.content
            return response
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            continue
    return "LLM Error: Cannot get response."


def prompt_differences(captions1: List[str], captions2: List[str]) -> str:
    caption1_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions1)]
    )
    caption2_concat = "\n".join(
        [f"Image {i + 1}: {caption}" for i, caption in enumerate(captions2)]
    )
    prompt = f"""Here are two groups of images:

Group 1:
```
{caption1_concat}
```

Group 2:
```
{caption2_concat}
```

What are the differences between the two groups of images?
Think carefully and summarize each difference in JSON format, such as:
```
{{"difference": several words, "rationale": group 1... while group 2...}}
```
Output JSON only. Do not include any other information.
"""
    return prompt


def get_differences(captions1: List[str], captions2: List[str], model: str) -> str:
    prompt = prompt_differences(captions1, captions2)
    differences = get_llm_output(prompt, model)
    try:
        differences = json.loads(differences)
    except Exception as e:
        logging.error(f"Difference Error: {e}")
    return differences


def test_get_llm_output():
    prompt = "hello"
    model = "gpt-4"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    model = "gpt-3.5-turbo"
    completion = get_llm_output(prompt, model)
    print(f"{model=}, {completion=}")
    # model = "vicuna"
    # completion = get_llm_output(prompt, model)
    # print(f"{model=}, {completion=}")


def test_get_llm_output_parallel():
    threads = []

    for _ in range(3):
        thread = threading.Thread(target=test_get_llm_output)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_get_differences():
    captions1 = [
        "A cat is sitting on a table",
        "A dog is sitting on a table",
        "A pig is sitting on a table",
    ]
    captions2 = [
        "A cat is sitting on the floor",
        "A dog is sitting on the floor",
        "A pig is sitting on the floor",
    ]
    differences = get_differences(captions1, captions2, "gpt-4")
    print(f"{differences=}")


if __name__ == "__main__":
    test_get_llm_output()
    test_get_llm_output_parallel()
