
import time
import json
import re
import numpy as np
import pandas as pd
import io
import sys
from flask import Flask, jsonify, request


def execute_python_code(code, local_path):
    """执行Python代码并获取最后的输出作为答案"""
    
    local_path_clean = local_path.rstrip('/')
    code = f"import os\nos.chdir('{local_path_clean}')\n" + code

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
            return None, "执行代码后没有任何输出。"

    except Exception as e:
        sys.stdout = old_stdout
        error_message = f"代码执行失败: {e}"
        return None, error_message


app = Flask(__name__)


@app.route('/python', methods=['POST'])
def python_code_exec():
    code_data = request.json
    print(code_data)
    try:
        result, error_message = execute_python_code(code_data['code'], local_path='/gemini/space/private/panchangzai/table2text/verl/data/table/MiMoTable_with_json')
        print("finished!")
        return jsonify({"result": result, "error_message": error_message}), 200
    except Exception as e:
        print("error:", e)
        return None, 400


# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

