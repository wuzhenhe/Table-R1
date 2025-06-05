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
import random
import pandas as pd
import datasets
import argparse

from utils.reward_score.MATH import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def read_json_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as fl:
        for line in fl:
            line = json.loads(line)
            # if line["level"] != "Level 1" and line["level"] != "Level 2":
            data.append(line)
    random.seed(2023)
    random.shuffle(data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_train_dir',
                        default='/gemini-1/space/space/private/pengjiaxin/projects/verl/data/math/raw/train.json')
    parser.add_argument('--local_test_dir',
                        default='/gemini-1/space/space/private/pengjiaxin/projects/verl/data/math/raw/test.json')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--save_dir', default="/gemini-1/space/space/private/pengjiaxin/projects/verl/data/math/math_500")

    args = parser.parse_args()

    data_source = 'lighteval/MATH'

    train_dataset = read_json_data(args.local_train_dir)[:500]
    test_dataset = read_json_data(args.local_test_dir)

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, example, idx):

        question = example['problem']
        question = question + '\n' + instruction_following

        # for qwen
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, and the final answer comes after </think>, <think> reasoning process here </think> answer here. Now the user asks you to solve a problem. After thinking, you finally reach a conclusion.\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>"""

        # for telechat
        # prefix = f"""<_system>You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<_user>\n{question}\n<_bot>\n<think>"""

        answer = example['solution']
        solution = extract_solution(answer)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prefix
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }

        print(data)

        return data


    train_dataset = [make_map_fn('train', line, idx) for idx, line in enumerate(train_dataset)]
    test_dataset = [make_map_fn('test', line, idx) for idx, line in enumerate(test_dataset)]

    save_dir = args.save_dir
    hdfs_dir = args.hdfs_dir

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


