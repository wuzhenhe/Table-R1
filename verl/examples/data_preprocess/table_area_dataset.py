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
import re
import argparse
import numpy as np
import io
import sys
import requests
import os
import matplotlib.pyplot as plt
from wrapt_timeout_decorator import *



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


def compare(list1, list2):
    # sort the list
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if np.isnan(list1[i]):
            if not np.isnan(list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False
    return True


def std_digit(list_nums):
    new_list = []
    for i in range(len(list_nums)):
        new_list.append(round(list_nums[i], 2))
    return new_list


def compute_general_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)


def compute_pie_chart_metric(references, predictions):
    processed_references = []
    processed_predictions = []
    for reference in references:
        if isinstance(reference, list):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)
    references = processed_references
    processed_references = []
    total = 0
    for reference in references:
        total += reference
    for reference in references:
        processed_references.append(round(reference / total, 2))

    for prediction in predictions:
        if isinstance(prediction, list):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    return compare(processed_references, processed_predictions)


def get_line_y_predictions(plt):
    line_y_predctions = []
    lines = plt.gca().get_lines()
    line_y_predctions = [list(line.get_ydata()) for line in lines]
    return line_y_predctions


def get_bar_y_predictions(plt):
    bar_y_predctions = []
    patches = plt.gca().patches
    bar_y_predctions = [patch.get_height() for patch in patches]
    return bar_y_predctions


def get_hbar_y_predictions(plt):
    hbar_y_predctions = []
    patches = plt.gca().patches
    hbar_y_predctions = [patch.get_width() for patch in patches]
    return hbar_y_predctions


def get_pie_y_predictions(plt):
    pie_y_predctions = []
    patches = plt.gca().patches
    for patch in patches:
        theta1, theta2 = patch.theta1, patch.theta2
        value = round((theta2 - theta1) / 360.0, 2)
        pie_y_predctions.append(value)
    return pie_y_predctions


def get_area_y_predictions(plt):
    area_y_predctions = []
    area_collections = plt.gca().collections
    for area_collection in area_collections:
        area_items = []
        for item in area_collection.get_paths()[0].vertices[:, 1]:
            if item != 0:
                area_items.append(item)
        area_y_predctions.append(area_items)
    return list(area_y_predctions)


def get_radar_y_predictions(plt):
    radar_y_predctions = []
    radar_lines = plt.gca().get_lines()
    radar_y_predctions = [list(line.get_ydata()) for line in radar_lines]
    for i in range(len(radar_y_predctions)):
        radar_y_predctions[i] = radar_y_predctions[i][:-1]
    return radar_y_predctions


def get_scatter_y_predictions(plt):
    scatter_y_predctions = []
    scatter_collections = plt.gca().collections
    for scatter_collection in scatter_collections:
        scatter_items = []
        for item in scatter_collection.get_offsets():
            scatter_items.append(item[1])
        scatter_y_predctions.append(scatter_items)
    return scatter_y_predctions


def get_waterfall_y_predictions(plt):
    waterfall_y_predctions = []
    patches = plt.gca().patches
    waterfall_y_predctions = [patch.get_height() for patch in patches]
    return waterfall_y_predctions


@timeout(15)
def execute(c):
    exec(c)

# ==== prediction Parsers ===


def parse_dp_prediction(prediction):
    pattern = r"Final Answer: (.+)"
    try:
        match = re.search(pattern, prediction)
        if match:
            return match.group(1)
        else:
            return ''
    except Exception as e:
        return ''


def parse_python_code(prediction):
    pattern = r"```python\n(.*?)```"
    try:
        matches = re.findall(pattern, prediction, flags=re.S)
        if matches:
            return matches[-1]
        else:
            return ''
    except Exception as e:
        return ''


def parse_code_output_prediction(prediction):
    pattern = r"Final Answer: (.+)"
    try:
        match = re.search(pattern, prediction)
        if match:
            return match.group(1)
        else:
            return ''
    except Exception as e:
        return ''


def build_chart_eval_code(sample):
    answer = sample['answer']
    chart_type = sample['chart_type']
    prediction = sample['response']

    python_code = parse_python_code(prediction)

    # TestCase
    eval_code = '''
if chart_type == 'line':
    y_predictions = get_line_y_predictions(plt)
if chart_type == 'bar':
    y_predictions = get_bar_y_predictions(plt)
if chart_type == 'hbar':
    y_predictions = get_hbar_y_predictions(plt)
if chart_type == 'pie':
    y_predictions = get_pie_y_predictions(plt)
if chart_type == 'area':
    y_predictions = get_area_y_predictions(plt)
if chart_type == 'radar':
    y_predictions = get_radar_y_predictions(plt)
if chart_type == 'scatter':
    y_predictions = get_scatter_y_predictions(plt)
if chart_type == 'waterfall':
    y_predictions = get_waterfall_y_predictions(plt)

if chart_type == 'pie':
    print(compute_pie_chart_metric(y_references, y_predictions))
else:
    print(compute_general_chart_metric(y_references, y_predictions))
    '''
    chart_eval_code = f'from chat_metric_utils import *\n{python_code}\n{answer}\nchart_type="{chart_type}"\n{eval_code}'
    if python_code == '':
        return '', ''
    return python_code, chart_eval_code


def pre_save_table_to_csv(table):
    table_json = []
    for item in table['data']:
        row_data = {}
        for i in range(len(table['columns'])):
            row_data[table['columns'][i]] = item[i]
        table_json.append(row_data)
    df = pd.DataFrame(table_json)
    df.to_csv('table.csv', index=False)


def surround_pycode_with_main(pycode):
    start_line = '''
if __name__ == '__main__':
'''
    pycode_lines = pycode.strip().split('\n')
    for line in pycode_lines:
        start_line += f'    {line}\n'
    return start_line


def parse_general_code_then_exec(prediction):
    ecr_1 = False
    python_code = parse_python_code(prediction)
    if python_code == '':
        return '', ecr_1
    try:
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            python_code = surround_pycode_with_main(python_code)
            execute(python_code)
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
        ecr_1 = True
    except Exception as e:
        output_value = ''
    if output_value != '':
        parsed_prediction = parse_code_output_prediction(output_value)
    else:
        parsed_prediction = ''
    return parsed_prediction, ecr_1


def parse_chart_code_then_exec(sample):
    ecr_1 = False
    python_code, chart_eval_code = build_chart_eval_code(sample)
    if python_code == '':
        return '', False
    try:
        python_code = surround_pycode_with_main(python_code)
        execute(python_code)
        ecr_1 = True
    except Exception as e:
        ecr_1 = False
    if ecr_1:
        pass
    try:
        from io import StringIO
        output = StringIO()
        stdout = sys.stdout
        try:
            sys.stdout = output
            chart_eval_code = surround_pycode_with_main(chart_eval_code)
            execute(chart_eval_code)
        finally:
            sys.stdout = stdout
        output_value = output.getvalue()
    except Exception as e:
        output_value = ''

    if output_value != '':
        parsed_prediction = output_value.strip()
    else:
        parsed_prediction = ''
    plt.close('all')
    return parsed_prediction, ecr_1


if __name__ == '__main__':
    random.seed(2023)
    save_dir = "/gemini/space/private/panchangzai/wzh/verl/data/table_area/0409"
    file_path = "/gemini/space/private/panchangzai/LLaMA-Factory-main-wzh/LLaMA-Factory-main/TableBench_TableArea/TableInstruct_TableArea/TableInstruct_TableArea_answer.jsonl"

    train_data_sources = ['table']
    test_data_sources = ['table']

    dataset = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # 解析每一行 JSON 数据
            data = json.loads(line)
            gold_area_match = re.findall(r"<gold_area>(.*?)</gold_area>", data["response"])
            if gold_area_match:
                data["gold_area"] = gold_area_match[-1].strip()
            else:
                print("area error.")

            data["ground_truth"] = {"answer": data["answer"], "gold_area": data["gold_area"], "instruction_type": data["instruction_type"], "qtype": data["qtype"]}
            data = {"instruction": data["instruction"], "ground_truth": data["ground_truth"]}

            dataset.append(data)

    random.shuffle(dataset)
    all_train_dataset = dataset[:int(0.95*len(dataset))]
    all_test_dataset = dataset[int(0.95*len(dataset)):]

    # add a row to each data item that represents a unique id
    def make_map_fn(split, example, idx):
        data_ = {
            "data_source": "table_area",
            "prompt": [{
                "role": "user",
                "content": example["instruction"]
            }],
            "ability": "table",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["ground_truth"]
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data_

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


