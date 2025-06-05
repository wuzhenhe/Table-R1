# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import time
import json
import re
import numpy as np
import pandas as pd
import io
import sys
import requests


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def execute_python_code(code):
    # python exec API Server
    BASE_URL = "http://10.244.23.142:5000"
    code_data={"code": code}
    max_try = 5
    while max_try>0:
        try:
            response = requests.post(f"{BASE_URL}/python", json=code_data)
            return response.json()['result'], response.json()['error_message']
        except Exception as e:
            max_try -= 1
            print("Warning! exec python api error.............\n{}".format(e))
            time.sleep(3)
    error_message = f"调用api失败"
    return None, error_message


def json_default(obj):
    """自定义 JSON 序列化函数"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def code_format_eval(prediction):
    code = extract_python_code(prediction)
    if code and "print(" in code and "pd.read_csv" in code:
        return True
    else:
        return False


def code_exec_result(prediction):
    prediction = extract_python_code(prediction)
    print("\nCleaned code:{}".format(prediction))
    result, error_message = execute_python_code(prediction)
    print("Exec result:{}".format(result))
    print("Exec error:{}".format(error_message))
    return result, error_message


def api_reward_model(question, model_output, ref_answer):
    from openai import OpenAI

    prompt = "你是一个评判助手，你的任务是根据问题和提供的标准答案来评估其他答案的正确性，判断的标准是，其他答案跟标准答案在关键结果上是否一致，" \
             "如果一致输出1，否则输出0，除此外不要输出其他内容。\n问题：{}\n标准答案：\n{}\n其他答案：\n{}"
    if "</think>" in model_output:
        content = prompt.format(question, ref_answer, model_output.split("</think>")[-1])
    else:
        content = prompt.format(question, ref_answer, model_output)

    openai_api_key = "sk-telenlp1234"
    openai_api_base = "http://10.244.14.132:8001/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, )

    def predict(query):
        max_try = 5
        max_tokens = 8192
        while max_try>0:
            try:
                model_name = client.models.list().data[0].id
                response = client.chat.completions.create(
                    model=model_name,
                    messages=query,
                    temperature=0.3,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    extra_body={
                        "repetition_penalty": 1.01,
                        "skip_special_tokens": False,
                        "spaces_between_special_tokens": False
                    },
                )
                max_try = -1
                return response.choices[0].message.model_dump()["content"]
            except:
                max_try -= 1
                time.sleep(60)
                print("Warning! reward api error.............")
        print("***inference failed***")
        return ''

    messages = [{"role": "user", "content": content}]
    answer = predict(messages)
    print("reward model output: {}".format(answer))
    return answer


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    positions['final_end'] = len(processed_str)-1
    positions['think_length'] = positions['think_end'] - positions['think_start']
    positions['answer_length'] = positions['final_end'] - positions['think_start']

    # Verify tag order
    # if positions['think_start'] > positions['think_end'] or \
    #         (positions['answer_length'] > positions['think_length']):
    if positions['think_start'] > positions['think_end']:
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    # check code format
    if validation_passed:
        processed_str = processed_str.split("</think>")[-1]
        validation_passed = code_format_eval(processed_str)

    return validation_passed


def compute_score(solution_str, ground_truth, prompt_str=None, format_reward=1) -> float:
    """
    get rule based score
    :param solution_str:
    :param ground_truth:
    :param prompt_str:
    :param format_reward:
    :return: Float score
    """
    solution_str = "<think>" + solution_str

    if "<｜Assistant｜>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<｜Assistant｜>")[0]
    elif "<_bot>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<_bot>")[0]
    elif "<|im_end|>" in prompt_str:
        question = prompt_str.split("输入问题：\n")[1].split("<|im_end|>")[0]
    else:
        question = prompt_str

    # Validate response structure
    format_correct = validate_response_structure(solution_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n[Question]\n{question}")
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")

    retval = 0
    if format_correct:
        try:
            code_str = solution_str.split("</think>")[-1].replace("<|im_end|>", "").replace("<｜end▁of▁sentence｜>", "").replace("<_end>", "").strip()
            exec_result, error_message = code_exec_result(code_str)
            if exec_result:
                if '1' in api_reward_model(question, exec_result, ground_truth):
                    retval = 2
                else:
                    retval = -1.5
            else:
                retval = -1.5

        except Exception as e:
            print(e)

    total_score = format_score + retval
    return total_score
