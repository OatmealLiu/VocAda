from my_agents.language import LLaMA3sAlpha

if __name__ == '__main__':

    llm = LLaMA3sAlpha(
        model='llama3-8b-instruct',
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=2048,
        do_sample=True
    )

    user_prompt = "Hello World!"
    system_prompt = "You are a helpful assistant"

    for i in range(3):
        reply = llm.infer(
            user_prompt,
            system_prompt=system_prompt
        )
        print(reply)
