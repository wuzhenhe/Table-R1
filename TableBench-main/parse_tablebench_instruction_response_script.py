import re
import os
import sys
from utils.file_util import read_json_file, write_json_to_file, iter_file_from_dir
from metrics.chart_metric_utils import *
import pandas as pd
import matplotlib.pyplot as plt
import ast
from wrapt_timeout_decorator import *
import json


@timeout(15)
# In case an infinite loop occurs
def execute(c):
    exec(c)

# ==== Prediction Parsers ===


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


def pre_save_table_to_csv(table):
    table_json = []
    for item in table['data']:
        row_data = {}
        for i in range(len(table['columns'])):
            row_data[table['columns'][i]] = item[i]
        table_json.append(row_data)
    df = pd.DataFrame(table_json)
    df.to_csv('table.csv', index=False)


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
    chart_eval_code = f'from metrics.chart_metric_utils import *\n{python_code}\n{answer}\nchart_type="{chart_type}"\n{eval_code}'
    if python_code == '':
        return '', ''
    return python_code, chart_eval_code


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
    # Fix some output without "Final Answer:"
    if parsed_prediction == '' and output_value != '':
        parsed_prediction = output_value.strip()

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

    parsed_prediction = {'True': True, 'False': False}.get(
        parsed_prediction, '')

    plt.close('all')
    return parsed_prediction, ecr_1


def parse_inference_results(inference_results):
    # === Start parsing ===
    parsed_results = []
    num = 0
    for sample in inference_results:
        #print(num)
        #num = num+1
        table = {"columns": sample['table.columns'], "data": sample['table.data']}
        #table = json.loads(sample['table'])
        prediction = sample['response']
        instruction_type = sample['instruction_type']
        if instruction_type == 'SCoT' or instruction_type == 'TCoT' or instruction_type == 'DP':
            qtype = sample['qtype']
            if qtype == 'Visualization':
                pre_save_table_to_csv(table)
                parsed_prediction, ecr_1 = parse_chart_code_then_exec(
                    sample)
                parsed_result = {
                    'parsed_prediction': parsed_prediction, 'ecr_1': ecr_1}
            else:
                parsed_prediction = parse_dp_prediction(prediction)
                parsed_result = {
                    'parsed_prediction': parsed_prediction}
        elif instruction_type == 'PoT':
            pre_save_table_to_csv(table)
            qtype = sample['qtype']
            if qtype == 'Visualization':
                parsed_prediction, ecr_1 = parse_chart_code_then_exec(
                    sample)
            else:
                if num!=1000:
                    parsed_prediction, ecr_1 = parse_general_code_then_exec(
                    prediction)
                else:
                    parsed_prediction=''
                    ecr_1=False
            parsed_result = {
                'parsed_prediction': parsed_prediction, 'ecr_1': ecr_1}
        # Process successful parsing ratio
        if parsed_prediction == '':
            parsed_result['Parse@1'] = False
        else:
            parsed_result['Parse@1'] = True
        # Save parsed result
        sample['parsed_result'] = parsed_result
        parsed_results.append(sample)
    return parsed_results


if __name__ == '__main__':
    # ==== Global settings ====
    PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXP_DIR = 'qwen25-3b'

    INFERENCE_RESULT_DIR = f'{PROJECT_ROOT_DIR}/model_results/{EXP_DIR}/inference_results'
    PARSED_RUSULT_DIR = f'{PROJECT_ROOT_DIR}/model_results/{EXP_DIR}/parsed_results'

    # ==== Load inference results ====
    for inference_result_file in iter_file_from_dir(f'{INFERENCE_RESULT_DIR}', '.jsonl'):

        print(f'Parsing {inference_result_file}')
        # === Load inference results ===
        inference_results = read_json_file(inference_result_file)
        if not isinstance(inference_results, list):
            inference_results = [inference_results]
        # === Parse inference results ===
        parsed_results = parse_inference_results(inference_results)
        # === Save parsed results ===
        write_json_to_file(
            f'{PARSED_RUSULT_DIR}/{os.path.basename(inference_result_file)}', parsed_results, is_json_line=True)
    print('Parsing completed.')
