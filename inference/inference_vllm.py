from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
import pandas as pd
import asyncio
import uuid
import time
import openpyxl
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import concurrent.futures



load_dotenv()
NUM = 0

class CustomReporterLLM:
    def __init__(self, model_name: str = None, config_file: str = None, topic: str = None):
        """
        支持动态选择模型，配置 API 密钥和基础 URL。
        Llama3.1-70B-Instruct
        Llama3-70B-Chat
        Llama3-8B-Chat
        CodeLlama-34B-Instruct
        TableLLM-Deepseek-Coder-7B
        """
        self.query = topic
        self.models = {
            "tableGPT2_7B": {
                "api_key": os.getenv("TABLEGPT2_7B_API_KEY"),
                "base_url": os.getenv("TABLEGPT2_7B_BASE_URL"),
            },
            "qwen2_72B": {
                "api_key": os.getenv("QWEN2_72B_API_KEY"),
                "base_url": os.getenv("QWEN2_72B_BASE_URL"),
            },
            "qwen25_32B": {
                "api_key": os.getenv("QWEN25_32B_API_KEY"),
                "base_url": os.getenv("QWEN25_32B_BASE_URL"),
            },
            "qwen25_coder": {
                "api_key": os.getenv("QWEN25_coder_API_KEY"),
                "base_url": os.getenv("QWEN25_coder_BASE_URL"),
            },
            "qwen25_qwq": {
                "api_key": os.getenv("QWEN25_QWQ_API_KEY"),
                "base_url": os.getenv("QWEN25_QWQ_BASE_URL"),
            },
            "qwen2.5-72b-instruct": {
                "api_key": os.getenv("QWEN25_72B_API_KEY"),
                "base_url": os.getenv("QWEN25_72B_BASE_URL"),
            },
            "mistral-arge-instruct-2407": {
                "api_key": os.getenv("MISTRAL_API_KEY"),
                "base_url": os.getenv("MISTRAL_BASE_URL"),
            },
            "deepseek-chat": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
            },
            "moonshot-v1-32k": {
                "api_key": "sk-zf8lAQmMqcyJUUlYcECl6GjwLGztFtl48RCwjbecObCjx72k",
                "base_url": "https://api.agicto.cn/v1",
            },
            "llama3-70b": {
                "api_key": os.getenv("LLAMA_3.3_70B_API_KEY"),
                "base_url": os.getenv("LLAMA_3.3_70B__BASE_URL"),
            },
            "qwen2-7B": {
                "api_key": "EMPTY",
                "base_url": "http://10.244.66.95:8000/v1",
            },
            "local": {
                "api_key": "EMPTY",
                "base_url": "http://127.0.0.1:8000/v1",
            },
        }

        if model_name not in self.models:
            raise ValueError(f"模型名称无效，请从以下列表中选择: {list(self.models.keys())}")

        self.model_name = 'openchat'
        self.api_key = self.models[model_name]["api_key"]
        self.base_url = self.models[model_name]["base_url"]

        # print(self.api_key, self.base_url)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def end2end_generate_report(self, prompt: str, is_json=False, temperature=0.0) -> str:
        """
        根据输入 Prompt 调用模型生成表格对应的报告。

        Args:
            temperature:
            is_json: json格式输出
            prompt (str): 输入的 Prompt。

        Returns:
            str: 模型生成的分类结果。
        """
        messages = [{"role": "system", "content": '你是一个助手'},
                    {"role": "user", "content": prompt}]

        try:
            if is_json:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    messages=messages,
                    response_format={'type': 'json_object'}
                )

            else:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    messages=messages
                )

            result = json.loads(completion.model_dump_json())
            return result['choices'][0]['message']['content']

        except Exception as e:
            logging.error(f"调用出错: {e}")
            return ''


class LLMInference:
    def __init__(self, llm: CustomReporterLLM):
        """
        初始化分类 Pipeline。

        Args:
            llm (DomainClassificationLLM): 领域分类模型对象。
        """
        self.llm = llm

    def inference(self, instruction: str):
        prompt = instruction
        generated_report = self.llm.end2end_generate_report(prompt)
        global NUM
        print(NUM, flush=True)
        NUM = NUM + 1
        return generated_report



def read_file_to_markdown(file_path):
    """
    根据文件路径读取文件，并返回其表格数据的字典表示。
    """
    _, file_extension = os.path.splitext(file_path)
    try:
        # Read file into a pandas DataFrame
        if file_extension.lower() == '.csv':
            try:
                df = pd.read_csv(file_path)
            except:
                df = pd.read_csv(file_path, encoding='gbk')
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    # Convert DataFrame to markdown
    return df.to_markdown(index=False).replace(' ', '')  # 删除markdown空格


def process_llms(input_path_file, output_path, model_name):
    """
    处理指定文件夹下的所有文件，并将结果保存为 Excel 文件。
    """
    results = []
    llm = CustomReporterLLM(model_name)
    pipeline = LLMInference(llm)

    with open(input_path_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    df = pd.json_normalize(data)  # 将JSON对象转换为DataFrame
    #df = df[0:10]
    instruction_list = list(df['instruction'])
    #type_list = list(df["instruction_type"])


    area_list = []

    '''for in_type, instruction in zip(type_list, instruction_list):
        print(in_type, ":", num)
        num = num+1
        generated_area_result = pipeline.inference(instruction)
        area_list.append(generated_area_result)'''
    df["response"] = 0
    #print(type_list[0])
    global NUM
    NUM = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(pipeline.inference, instruction) for instruction in instruction_list]
        # 创建一个字典，将 future 和它的索引关联起来
        index_map = {future: idx for idx, future in enumerate(futures)}
        # 按照完成顺序处理结果，但根据索引重新排序
        results = [None] * len(instruction_list)
        for future in concurrent.futures.as_completed(futures):
            idx = index_map[future]
            results[idx] = future.result()

        area_list = results

    df["response"] = area_list



    # 将DataFrame转换为JSONL格式并保存
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    folder_path ="/gemini/space/private/inference"
    model_name = "local"
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            input_file_path = os.path.join(folder_path, filename)
            # 生成输出文件名
            base_name = os.path.splitext(filename)[0]
            print(base_name)
            #if "PoT" not in base_name:
                #continue
            output_filename = f"{base_name}_output.jsonl"
            output_dic = '/gemini/space/private/LLaMA-Factory-main/TableBench_TableArea/TableBench-main_v2/model_results/llama3.1-8b-rl900/inference_results'
            os.makedirs(output_dic, exist_ok=True)
            output_file_path = os.path.join(output_dic, output_filename)
            #output_file_path = os.path.join(folder_path + '/output', output_filename)
            # 执行 process 函数
            #print(input_file_path)
            #print(output_file_path)
            process_llms(input_file_path, output_file_path,model_name)
    print("Over")
