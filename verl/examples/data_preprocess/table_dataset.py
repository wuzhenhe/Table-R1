# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import json
import time
import random
import pandas as pd
import datasets


import argparse


def add_prompt(table_desc, query):
        prompt_head = """
你是一位精通 Python 的数据分析师。你的任务是编写可执行的 Python 代码来解析表格，然后回答问题。

要求：
1. 根据问题，写出你的分析思路和方法，再根据这种方法写 Python 代码。
2. 请严格按照给你的文件路径和表格描述来生成代码。
3. 只生成一个代码块，并且严格以 ```python 开始，并以 ``` 结束。
4. 你的分析必须完全基于表格数据。如果用户的问题与数据分析无关，请礼貌拒绝。
5. 你需要生成可执行代码。如果有结果需要展示，请将结果存入answer函数中， 并用print展示。
6. 确保使用python库中pd.read_csv函数，读取给你的表格文件路径来进行数据处理，如需要读取多个表格，注意使用文件路径进行变量名区分。
7. 生成代码的过程中，请不要将数据转成DataFrame格式，请务必用pd.read_csv函数进行表格内容读取。

以下是提供的表格信息：
"""

        prompt_tail = """
确保最终答案是 Python 代码的最后一行，并且只能以 print(f'{answer}') 的形式出现，不能有其他形式。


让我们逐步思考，然后生成 Python 代码来分析表格并展示问题的最终答案。
输入问题：
"""

        query = prompt_head + "\n" + table_desc + "\n" + prompt_tail + query
        return query


# def add_prompt(table_desc, query):
#     prompt_head = """
# 你是一位精通 Python 的数据分析师，你的任务是通过给你的表格信息和问题编写可执行的 Python 代码来解析表格并回答问题。
#
# 要求：
# 1. 根据问题，写出你的分析思路和方法，再根据这种方法写 Python 代码。
# 2. 请严格按照给你的文件路径和表格描述来生成代码，只生成一个代码块，并且严格以 ```python 开始，并以 ``` 结束。
# 3. 注意理解表格结构，尤其是注意表头可能比较复杂需要小心处理。
# 4. 你的分析必须完全基于表格数据。如果用户的问题与数据分析无关，请礼貌拒绝。
# 5. 你需要生成可执行代码，将最终answer用print函数打印出来。
# 6. 确保使用python库中pd.read_csv函数来读取给你的表格文件路径，如需要读取多个表格，注意使用文件路径进行变量名区分。
#
# 以下是提供的表格信息：
# """
#
#     prompt_tail = """
#
# 让我们think step by step，然后生成 Python 代码来分析表格并展示问题的最终答案。
# 输入问题：
# """
#
#     query = prompt_head + "\n" + table_desc + "\n" + prompt_tail + query
#     return query


def change_data_distribution(data):
    new_data = []
    for line in data:
        if line['table_difficulty'] == 'medium':
            new_data += [line]*2
        elif line['table_difficulty'] == 'easy':
            new_data += [line]*3
        else:
            new_data.append(line)
    return new_data


def read_json_data(file, source):
    if source=='normal_reasoning':
        data = []
        with open(file, 'r', encoding='utf-8') as fl:
            for line in fl:
                line = json.loads(line)
                line['problem'] = line['model_input'][0]['content']
                line['solution'] = line['reference_answer']
                line['category'] = source
                data.append(line)

    elif source=='table':
        all_data = []
        with open(file, 'r', encoding='utf-8') as fl:
            for line in fl:
                line = json.loads(line)
                if len(line['table_desc']) < 10:
                    print([line['table_desc']])
                    continue

                # prefix = "/gemini-1/space/space/private/pengjiaxin/projects/Logic-RL/data/table/MiMoTable_with_json/"
                # line['table_desc'] = line['table_desc'].replace("文件路径: ", "文件路径: " + prefix)

                table_desc, query = line['table_desc'], line['question']

                query = add_prompt(table_desc, query)

                line['problem'] = query
                line['solution'] = line['gold_truth']
                line['category'] = source
                all_data.append(line)
    else:
        data = []
        with open(file, 'r', encoding='utf-8') as fl:
            for line in fl:
                line = json.loads(line)
                line['category'] = source
                data.append(line)

    return all_data


if __name__ == '__main__':
    random.seed(2023)
    save_dir = "/gemini/space/private/panchangzai/table2text/verl/data/table/table_v10327_template0_easier"
    local_test_dirs = ['/gemini/space/private/panchangzai/table2text/verl/data/table/raw/v1_0327/table_rl_20250327_300_test.jsonl']
    local_train_dirs = ['/gemini/space/private/panchangzai/table2text/verl/data/table/raw/v1_0327/table_rl_20250327_1813_train.jsonl']

    train_data_sources = ['table']
    test_data_sources = ['table']

    all_train_dataset = []
    for f, source in zip(local_train_dirs, train_data_sources):
        train_dataset = read_json_data(f, source)
        all_train_dataset.extend(train_dataset)
    all_test_dataset = []
    for f, source in zip(local_test_dirs, test_data_sources):
        test_dataset = read_json_data(f, source)
        all_test_dataset.extend(test_dataset)

    random.shuffle(all_train_dataset)
    random.shuffle(all_test_dataset)

    # 改变table难度分布
    all_train_dataset = change_data_distribution(all_train_dataset)
    print(len(all_train_dataset), len(all_test_dataset))

    # if there is no test data, get test data from train data
    if len(all_test_dataset) == 0:
        all_test_dataset = all_train_dataset[:100]
        all_train_dataset = all_train_dataset[100:]

    # add a row to each data item that represents a unique id
    def make_map_fn(split, example, idx):
        question = example['problem']

        # for telechat
        prefix = f"""<_system>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<_user>\n{question}\n<_bot>\n<think>"""
        # telechat template_2: only have <think> without <answer>
        prefix = f"""<_system>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, and then the answer comes after the reasoning process, <think> reasoning process here </think> answer here. Now the user asks you to solve a logical reasoning problem. After deep thinking, you finally reach a conclusion.\n<_user>\n{question}\n<_bot>\n<think>"""
        # for qwen: only have <think> without <answer>
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, and the final answer comes after </think>, <think> reasoning process here </think> answer here. Now the user asks you to solve a problem. After thinking, you finally reach a conclusion.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
        # for r1-distill-qwen: only have <think> without <answer>
        # prefix = f"""<｜begin▁of▁sentence｜>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, and the final answer comes after </think>, <think> reasoning process here </think> answer here. Now the user asks you to solve a problem. After thinking, you finally reach a conclusion.\n<｜User｜>\n{question}\n<｜Assistant｜><think>"""

        solution = example['solution']

        data = {
            "data_source": example['category'],
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "table",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }

        # print(data)

        return data

    train_dataset = [make_map_fn('train', line, idx) for idx, line in enumerate(all_train_dataset)]
    test_dataset = [make_map_fn('test', line, idx) for idx, line in enumerate(all_test_dataset)]

    print(len(train_dataset), len(test_dataset))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"Directory '{save_dir}' created successfully")
    else:
        print(f"Directory '{save_dir}' already exists")

    train_dataset = pd.DataFrame(train_dataset)
    test_dataset = pd.DataFrame(test_dataset)

    print(train_dataset)

    train_dataset.to_parquet(os.path.join(save_dir, 'train.parquet'), index=0)
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'), index=0)


