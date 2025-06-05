import time
import sys
import io
import re
import pandas as pd
import json
import random
from tqdm.contrib import tzip
from tqdm import tqdm
import traceback


def add_prompt(table_desc, query):
    """
    0
    :param table_desc:
    :param query:
    :return:
    """
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


def add_table_desc(line):
    new_data = []
    for data in line['table_desc'].split("====="):
        data = data.strip()
        if data:
            table_path = data.split("文件路径: ")[1].split("\n")[0].strip()
            try:
                df = pd.read_csv(table_path)
            except:
                print(table_path)
                continue

            # data = data.split("列描述:")[0]
            rows_desc = []
            rows_desc.append({"第1行": df.columns.to_list()})
            for i in range(min(9, df.shape[0])):
                row = df.iloc[i, :].to_list()
                rows_desc.append({"第{}行".format(i+2):row})

            data += "\n这是前10行的数据展示：\n{}".format(rows_desc)

            columns_desc = []
            columns = df.columns.to_list()
            for i in range(df.shape[1]):
                column = [columns[i]] + df.iloc[:4, i].to_list()
                columns_desc.append({"第{}列".format(i+1): column})

            data += "\n这是每列前5行的数据展示：\n{}".format(columns_desc)

            new_data.append(data)

    new_data = "\n=====\n".join(new_data)
    line['table_desc'] = new_data
    return line


def process_data(datas):
    data = []
    for line in datas:
        prefix = "MiMoTable_with_json/"
        line['table_desc'] = line['table_desc'].replace("文件路径: ", "文件路径: " + prefix)
        # change tabel desc
        # line = add_table_desc(line)
        table_desc, query = line['table_desc'], line['question']
        # add prompt
        query = add_prompt(table_desc, query)

        line['problem'] = query
        line['solution'] = line['gold_truth']
        data.append(line)

    return data


def extract_python_code(prediction):
    """使用正则表达式提取三反引号包裹的Python代码"""
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, prediction, re.DOTALL)
    code = "\n".join(code_blocks)
    return code


def execute_python_code(code):
    """执行Python代码并获取最后的输出作为答案"""
    try:
        # 重定向标准输出
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # 执行代码
        local_vars = {}
        exec(code, local_vars)

        # 获取标准输出结果
        output = buffer.getvalue().strip()
        sys.stdout = old_stdout

        if output:
            return output, None
        else:
            error_message = {
                "error_type": None,
                "error_message": "执行代码后没有打印出任何输出。",
                "stack_trace": None,
                "executed_code": code
            }
            return None, error_message

    except Exception as e:
        sys.stdout = old_stdout
        # 捕获详细的错误信息，包括异常类型、异常消息、堆栈跟踪信息
        error_message = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "stack_trace": traceback.format_exc(),
            "executed_code": code
        }
        return None, error_message


def code_exec_result(prediction):
    prediction = extract_python_code(prediction)
    # print("\nCleaned code:{}".format(prediction))
    result, error_message = execute_python_code(prediction)
    print("Exec result:{}".format(result))
    print("Exec error:{}\n\n\n".format(error_message))
    print("=============================")
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
    openai_api_base = "http://10.30.129.200:25242/v1"
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


def qwen32_ins_api(query):
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://10.244.77.123:8358/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, )
    print("model in:\n{}\n\n".format(query))

    def predict(query):
        max_try = 5
        max_tokens = 16000
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
                time.sleep(10)
                print("Warning! llm api error.............")
        print("***inference failed***")
        return ''

    messages = [
            {"role": "user", "content":query},
            ]
    answer = predict(messages)

    print("model output:\n{}".format(answer))
    return answer


def qwq_api(content):
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://10.244.109.95:8357/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, )
    print("model in:\n{}\n\n".format(content))

    def predict(query):
        max_try = 5
        max_tokens = 20000
        while max_try>0:
            try:
                model_name = client.models.list().data[0].id
                response = client.chat.completions.create(
                    model=model_name,
                    messages=query,
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    extra_body={
                        "repetition_penalty": 1.0,
                        "skip_special_tokens": False,
                        "spaces_between_special_tokens": False
                    },
                )
                max_try = -1
                return response.choices[0].message.model_dump()["content"]
            except Exception as e:
                max_try -= 1
                time.sleep(10)
                print("Warning! llm api error.............{}".format(e))
        print("***inference failed***")
        return ''

    messages = [{"role": "user", "content": content}]
    answer = predict(messages)

    print("model output:\n{}".format(answer))
    return answer.split("</think>")[1]


in_file='xxx/verl/data/table/raw/v1_0327/table_rl_20250327_300_test.jsonl'
out_file = "xxx/verl/data/table/table_rl_20250327_300_test_qwen32_rl_step100_prompt0_output.jsonl"

data = []
with open(in_file, 'r', encoding='utf-8') as fl:
    for line in fl:
        line = json.loads(line)
        data.append(line)

random.seed(2023)
random.shuffle(data)
test_data = data
test_data = process_data(test_data)

res, code_res = [], []
for idx, line in enumerate(test_data):
    print("idx:{}".format(idx))
    llm_output = qwen32_ins_api(line['problem'])
    code_result, error_message = code_exec_result(llm_output)
    res.append(llm_output)
    code_res.append(code_result)


wl = open(out_file, 'w', encoding='utf-8')
data = []
for cr, r, line in zip(code_res, res, test_data):
    if cr:
        if '1' in api_reward_model(line['question'], cr, line['gold_truth']):
            retval = 1
        else:
            retval = 0
    else:
        retval = 0

    print("label:", retval)
    line['model_output'] = r
    line['code_exec_res'] = cr
    line['label'] = retval
    print(json.dumps(line, ensure_ascii=False), file=wl)
    wl.flush()
    data.append(line)

    print("\n===============\n")


easy_table = [line for line in data if line['table_difficulty'] == 'easy']
medium_table = [line for line in data if line['table_difficulty'] == 'medium']
hard_table = [line for line in data if line['table_difficulty'] == 'hard']

exec_error_data = [line for line in data if line['code_exec_res'] == None]
print("exec error acc:")
print(len(exec_error_data), len(data), len(exec_error_data) / len(data))

print("all acc:")
right_data=[line for line in data if line['label']==1]
print(len(right_data), len(data), len(right_data)/len(data))

if len(easy_table)>0:
    print("easy table acc:")
    right_data = [line for line in easy_table if line['label'] == 1]
    print(len(right_data), len(easy_table), len(right_data) / len(easy_table))
if len(medium_table)>0:
    print("medium table acc:")
    right_data = [line for line in medium_table if line['label'] == 1]
    print(len(right_data), len(medium_table), len(right_data) / len(medium_table))
if len(hard_table):
    print("hard table acc:")
    right_data = [line for line in hard_table if line['label'] == 1]
    print(len(right_data), len(hard_table), len(right_data) / len(hard_table))

print("\n=======================\n")



