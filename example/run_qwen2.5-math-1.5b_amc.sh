set -x
ray stop --force
export VLLM_ATTENTION_BACKEND=XFORMERS
# export WANDB_MODE="offline"
# export WANDB_API_KEY='Your Wandb API Key'
export SWANLAB_API_KEY='Your Swanlab API Key'
export SWANLAB_LOG_DIR='Your Swanlab Log Directory'

vbs=7384
tbs=16
prompt_length=1024
response_length=2048
mbs=256
kl=0.001
n=16
t=0.6
topp=1.0
pt=0.9
ptopp=1.0
proposer_interval=1
repetition_penalty=1
reward_baseline=0.5
proposer_train_metric="val_reward"
save_load_interval=15

exp_name="qwen2.5-math-1.5b_amc"

project_name='CPMobius'

CKPT_DIR="Your Checkpoint Directory"
mkdir -p ${CKPT_DIR}

ray start --head --port=6789



python3 -Xfrozen_modules=off -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=Your path to amc_x10.parquet \
    data.train_batch_size=${tbs} \
    data.val_batch_size=${vbs} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Your path to Qwen2.5-Math-1.5B \
    actor_rollout_ref.model.proposer_path=Your path to coach model \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mbs} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=28000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.proposer_entropy_coeff=0.01 \
    actor_rollout_ref.actor.entropy_coeff=-0.01 \
    +actor_rollout_ref.actor.remove_previous_ckpt=False \
    +actor_rollout_ref.actor.proposer_train_metric=${proposer_train_metric} \
    +actor_rollout_ref.actor.val_use_rm=False \
    +actor_rollout_ref.actor.val_do_sample=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=28000 \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.top_p=${topp} \
    actor_rollout_ref.rollout.temperature=${t} \
    +actor_rollout_ref.rollout.proposer_top_p=${ptopp} \
    +actor_rollout_ref.rollout.proposer_temperature=${pt} \
    +actor_rollout_ref.rollout.repetition_penalty=${repetition_penalty} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=False \
    reward_model.model.path=Qwen/Qwen2.5-Math-PRM-7B \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.use_dynamic_bsz=True \
    reward_model.forward_max_token_len_per_gpu=27000 \
    reward_model.model.use_remove_padding=False \
    +reward_model.model.baseline=${reward_baseline}\
    +reward_model.model.is_qwen_prm=True \
    +reward_model.model.is_token_level=False \
    +reward_model.model.trust_remote_code=True \
    +reward_model.format_penalty=True \
    +reward_model.format_text="\\boxed" \
    +reward_model.format_penalty_value=-1 \
    +reward_model.use_best_of_n=True \
    algorithm.kl_ctrl.kl_coef=${kl} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1000 \
    trainer.task=gan \
    +trainer.reward_manager_batched=True \
    +trainer.filter_lower_bound=0.2 \
    +trainer.filter_upper_bound=0.8 \
    +trainer.proposer_update_interval=${proposer_interval} \
    +trainer.save_load_actor=True \
    +trainer.save_load_interval=${save_load_interval} \
    +trainer.save_load_begin_steps=100 $@
    
