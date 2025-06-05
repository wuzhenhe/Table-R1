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
import os
from wrapt_timeout_decorator import *
import evaluate
import string
import math


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


'''def get_line_y_predictions(plt):
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
    return waterfall_y_predctions'''


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


'''def parse_chart_code_then_exec(sample):
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
    return parsed_prediction, ecr_1'''


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def prepsocess(references,predictions):
    processed_predictions = []
    processed_references = []
    for i in range(len(predictions)):
        prediction = predictions[i]
        reference = references[i]
        # normalize prediction and reference
        prediction = normalize_answer(prediction)
        reference = normalize_answer(reference)
        if len(prediction) == 0:
            prediction = '#'
            processed_predictions.append(prediction)
            processed_references.append(reference)
        else:
            processed_predictions.append(prediction)
            processed_references.append(reference)
    predictions = processed_predictions
    references = processed_references
    return references,predictions


def calculate_iou(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)

    # 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # 计算 IOU
    if len(union) == 0:
        return 0.0  # 如果两个列表都为空，IOU 为 1
    return float(len(intersection)) / float(len(union))


def check_numbers(a, b):
    # 尝试将输入转换为浮点数
    try:
        num_a = float(a)
        num_b = float(b)
    except ValueError:
        return 0  # 如果转换失败，说明不是数字，直接返回0
    if pd.isnull(num_a) or pd.isnull(num_b):
        return 0
    if math.isnan(num_a) or math.isnan(num_b):
        return 0

    # 获取两个数字的整数部分（向下取整）
    try:
        int_a = int(num_a // 1)
        int_b = int(num_b // 1)
    except ValueError:
        return 0

    # 判断整数部分是否相等
    if int_a != int_b:
        return 0

    # 计算两个数字的差值占较小值绝对值的比例（避免除以0的情况）
    diff = abs(num_a - num_b)
    min_abs = min(abs(num_a), abs(num_b))
    if min_abs == 0:
        # 如果两个数字都是0，则认为相差0%，返回1
        if diff == 0:
            return 1
        else:
            return 0
    else:
        # 判断相差是否小于1%
        percent_diff = diff / min_abs
        return 1 if percent_diff < 0.01 else 0


def calculate_score1(get_area, gold_area):
    if get_area == "":
        return 0
    # 计算column分数
    match_gold = re.search(r'"columns":\s*(\[.*?\])', gold_area, re.DOTALL)
    match_get = re.search(r'"columns":\s*(\[.*?\])', get_area, re.DOTALL)
    if match_gold:
        try:
            gold_columns_str = match_gold.group(1)
            gold_columns_list = json.loads(gold_columns_str)
        except json.JSONDecodeError as e:
            gold_columns_list = []
    else:
        gold_columns_list = []
    if match_get:
        try:
            get_columns_str = match_get.group(1)
            get_columns_list = json.loads(get_columns_str)
        except json.JSONDecodeError as e:
            get_columns_list = []
    else:
        get_columns_list = []
    score_columns = calculate_iou(gold_columns_list, get_columns_list)
    # 计算row分数
    match_gold = re.search(r'"rows":\s*(\[.*?\])', gold_area, re.DOTALL)
    match_get = re.search(r'"rows":\s*(\[.*?\])', get_area, re.DOTALL)
    if match_gold:
        try:
            gold_rows_str = match_gold.group(1)
            gold_rows_list = json.loads(gold_rows_str)
        except json.JSONDecodeError as e:
            gold_rows_list = []
    else:
        gold_rows_list = []
    if match_get:
        try:
            get_rows_str = match_get.group(1)
            get_rows_list = json.loads(get_rows_str)
        except json.JSONDecodeError as e:
            get_rows_list = []
    else:
        get_rows_list = []
    score_rows = calculate_iou(gold_rows_list, get_rows_list)
    return 0.5*(score_columns + score_rows)


def calculate_score2(parsed_result, answer):
    '''references, predictions = prepsocess([answer], [parsed_result['parsed_prediction']])
    rouge = evaluate.load('/gemini/space/private/panchangzai/wzh/verl/verl/utils/rouge')
    rouge_score = rouge.compute(references=references, predictions=predictions)
    rouge_L = round(rouge_score['rougeL'], 2)
    if 'ecr_1' not in parsed_result:
        return rouge_L
    else:
        score = 0
        if parsed_result['ecr_1']:
            score = score + 0.1
            score = score + 0.9 * rouge_L
            return score
        else:
            return 0'''
    if parsed_result['parsed_prediction'] == answer:
        return 1
    else:
        # 如果是相差很小的数字，也返回1
        if check_numbers(answer, parsed_result['parsed_prediction']):
            return 1
        return 0


def check_repeated_segments(text, min_length=5, threshold=5):
    """
    检查字符串末尾是否有大量重复的字符段。

    参数:
        text (str): 输入的字符串。
        min_length (int): 最小重复子字符串的长度，默认为3。
        threshold (int): 重复次数的阈值，默认为3。

    返回:
        bool: 如果末尾存在大量重复的字符段，返回True，否则返回False。
        str: 重复的字符段（如果有）。
    """
    if len(text) < min_length * threshold:
        # 如果字符串长度小于最小重复长度乘以阈值，直接返回False
        return False, ""

    # 从后向前检查重复的子字符串
    for length in range(min_length, len(text) // threshold + 1):
        # 提取末尾的子字符串
        substring = text[-length:]
        # 检查是否重复
        if text.endswith(substring * threshold):
            return True, substring

    return False, ""



def compute_score(solution_str, ground_truth, prompt_str=None, step=0, format_reward=1) -> float:
    """
    get rule based score
    :param step:
    :param solution_str:
    :param ground_truth:
    :param prompt_str:
    :param format_reward:
    :return: Float score
    """
    # 拆分ground truth
    answer = ground_truth["answer"]
    gold_area = ground_truth["gold_area"]
    instruction_type = ground_truth["instruction_type"]
    #qtype = ground_truth["qtype"]
    # 提取答案
    parsed_result = {}
    if instruction_type == 'SCoT' or instruction_type == 'TCoT' or instruction_type == 'DP':
        parsed_prediction = parse_dp_prediction(solution_str)
        parsed_result = {'parsed_prediction': parsed_prediction}
    elif instruction_type == 'PoT':
        parsed_prediction, ecr_1 = parse_general_code_then_exec(solution_str)
        parsed_result = {'parsed_prediction': parsed_prediction, 'ecr_1': ecr_1}

    # 提取gold area
    get_area = ""
    gold_area_match = re.findall(r"<gold_area>(.*?)</gold_area>", solution_str)
    if gold_area_match:
        get_area = gold_area_match[-1].strip()
    
    # 计算区域得分
    score1 = calculate_score1(get_area, gold_area)
    score2 = calculate_score2(parsed_result, answer)
    # 检查是否有大量重复生成
    is_repeated, _ = check_repeated_segments(solution_str)
    if is_repeated is True:
        score2 = 0
    step_hundred = math.floor(step / 100)
    #a = 0.3 * np.exp(-0.183 * step_hundred)  # (0,0.3) (600,0.1) batch_size=32
    a = 0.3 * np.exp(-0.0915 * step_hundred)  # (0,0.3) (1200,0.1) batch_size=16
    a = 0.1
    total_score = a*score1+(1-a)*score2
    #total_score = 0.1 * score1 + 0.9 * score2
    #total_score = score2
    
    return total_score, a*score1, (1-a)*score2
