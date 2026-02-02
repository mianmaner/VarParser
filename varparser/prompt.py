from openai import OpenAI
from groq import Groq
import json
import yaml
import os
import re

# Setting with your OpenAI API
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = config['model']
client = OpenAI(api_key=config['api_key'], base_url=config['base_url'])
# client = Groq(api_key=config['api_key'])


def llm_querying(query_log, var_prompt, reference, temperature=0.0):
    instruction = "As a log parsing expert, refer to variable examples, replace variables with the corresponding label (no spaces in curly brackets) in the log if present. Just return only the parsed log in backtick and ensure it matches exactly"

    prompt_message = ""
    if len(var_prompt) > 0:
        prompt_message += "Variable Examples: "
        var_list = []
        for label, var in var_prompt.items():
            var_list.append(f"'{var}' -> {{{label}}}")
        prompt_message += ", ".join(var_list) + "\n"
    prompt_message += f"Format Reference: {reference}\n"

    prompt_message += f'`{query_log}`'

    # print(prompt_message)
    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": prompt_message}]

    # print(prompt_message)

    # messages.append({"role": "user", "content": prompt_message})
    # messages.append(
    #     {"role": "assistant", "content": "Okay, my following response will only be the log where variables are replaced by their corresponding labels."})

    # messages.append({"role": "user", "content": f"{logs}"})
    # print(messages)
    retry_times = 0
    while retry_times < 10:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=False
            )
            tokens = response.usage.total_tokens
            response = response.choices[0].message.content.strip('`')
            return response, tokens
        except Exception as e:
            print("Exception :", e)
            retry_times += 1
    return
