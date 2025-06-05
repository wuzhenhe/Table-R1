#!/bin/bash


cd /

nohup python -m vllm.entrypoints.openai.api_server --model  /gemini/space/private/llama3.1-8b-rl900 --served-model-name "openchat" --max_model_len 8192 --tensor-parallel-size 8 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes >./llama3.1-8b-rl900_tablebench.log 2>&1 &

sleep 200

python inference_vllm.py
pkill -f vllm