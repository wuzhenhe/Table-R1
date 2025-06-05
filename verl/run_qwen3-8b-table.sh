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
