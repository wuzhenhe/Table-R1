# Table-R1

## RE-SFT
We implement RE-SFT using the latest version of LLaMa-Factory (up to May 10, 2025).
Install according to the official documentation.
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
We register the TableInstruct-RE dataset in LLaMa-Factory/data/dataset_info.json, and provide the TableInstruct-RE as TableInstruct_TableArea.json.
```python
 "table_bench_all": {
    "file_name": "TableInstruct_TableArea.json",
    "columns": {
      "prompt": "prompt",
      "response": "response"
  }
```
We offer a training file for reference at LLaMA-Factory-main/examples/train_full/qwen3_8b_full_sft.yaml:
```python
model_name_or_path: /gemini/space/private/model/Qwen3-8B-Instruct/
#trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]

### dataset
dataset: table_bench_all  
template: qwen3  
cutoff_len: 4096
max_samples: 19661 
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /gemini/space/private/LLaMA-Factory-main/saves/qwen3-8b/full/sft 
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 64
learning_rate: 2.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```
If using a machine with 8*A100, execute the training command as follows:
```bash
export WANDB_DISABLED=true
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen3_8b_full_sft_origin.yaml
```

## TARPO
We use the verl (v0.2) framework for reinforcement learning. Please install the environment following the official instructions.
To adapt to Qwen3, additional package upgrades are also required:
```bash
pip install vllm transformers tensordict --upgrade
```
We provide the TableInstruct-RE dataset in the format required for verl training at data/table_area/0409.
We offer a training file for reference at verl/run_qwen3-8b-table.sh:
```python
set -x
export HYDRA_FULL_ERROR=1
#export CUDA_VISIBLE_DEVICES=1,2,3,4
export VLLM_USE_V1=1

MODEL_PATH=/nvfile-heatstorage/nlp/private/saves/qwen3-8b/full/sft
MODEL_SAVE_DIR=/nvfile-heatstorage/nlp/private/saves/qwen3-8b-tarpo

#export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/nvfile-heatstorage/nlp/private/verl/data/table_area/0409/train.parquet \
    data.val_files=/nvfile-heatstorage/nlp/private/verl/data/table_area/0409/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=11000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.grad_clip=0.9 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.98 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=naive \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='TableBench_rl' \
    trainer.experiment_name='qwen3_8b_sft_verl_tarpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$MODEL_SAVE_DIR \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@ 2>&1 | tee qwen3_8b_tarpo.log
```
If using a machine with 4*H100, execute the training command as follows:
```bash
nohup sh run_qwen3-8b-table_tarpo.sh > test.log 2>&1 &
```
execute the training command to combine parameters as follows:
```
cd verl/scripts
python model_merger.py --local_dir /nvfile-heatstorage/nlp/private/saves/qwen3-8b-tarpo/global_step_1000/actor
```

## Vllm Inference
Modify the addresses and execution parameters in inference_vllm.py and inference_vllm.sh according to your actual situation. Execute the command to inference:
```bash
cd inference
nohup sh inference_vllm.sh > output.log 2>&1 &
```
We provide the processed TableBench data with added table regions in the inference/ folder.

## Evaluate
Update the paths to the inference results files in parse_tablebench_instruction_response_script.py and eval_tablebench_script.py for evaluation. Here’s a reference example:
```bash
cd TableBench-main
export PYTHONPATH="/gemini/space/private/TableBench-main:$PYTHONPATH"
python parse_tablebench_instruction_response_script.py
python eval_tablebench_script.py
```