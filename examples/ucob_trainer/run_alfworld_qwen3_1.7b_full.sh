ENGINE=vllm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env.sh"

set -x
UCOB_REFLECTION_BACKEND=vllm

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

TRAIN_PARQUET="${VERL_AGENT_DATA_DIR}/text/train.parquet"
VAL_PARQUET="${VERL_AGENT_DATA_DIR}/text/test.parquet"

num_cpus_per_env_worker=0.1

# UCOB/GiGPO training knobs. Legacy teacher loss stays off in the full setting.
teacher_loss_enable=false
legacy_teacher_coef=0.0
rollout_mode=mixed
adaptive_enable=false
topk_enable=false
action_loss_mask_enable=false
require_thinking=false
entropy_coeff=0.00
rollout_top_p=1.0
rollout_max_model_len=10240
gamma=0.95
gigpo_step_advantage_w=1.0
gigpo_mode=mean_norm

# CBSD same-anchor comparison. The stronger branch distills its action into the
# opposite prompt context.
cbsd_enable=true
cbsd_coef=0.1
cbsd_action_mask_enable=true
cbsd_margin=0.05
cbsd_tau=0.2
cbsd_weight_max=2.0
cbsd_max_pairs_per_group=2
cbsd_source_branches='[good,none]'
cbsd_target_policy=opposite
cbsd_loss_type=topk_opd
cbsd_opd_granularity=turn
cbsd_opd_scope=response
cbsd_topk_k=32
cbsd_gate_enable=true
cbsd_gate_beta=5.0
cbsd_topk_norm_to_one=true
cbsd_topk_clip_log_ratio=false
cbsd_action_weight=2.0
cbsd_max_turns_per_trajectory=0

# Skill-memory retrieval and writing.
skillmem_top_k=3
skillmem_eval_top_k=3
skillmem_dual_pool_retrieval=true
skillmem_memory_pool_mode=both
skillmem_task_top_k=3
skillmem_state_top_k=3
skillmem_eval_task_top_k=3
skillmem_eval_state_top_k=3
skillmem_alpha=0.1
skillmem_beta=0.2
skillmem_task_utility_beta=0.2
skillmem_state_utility_beta=0.2
skillmem_ucb_scale=0.1
skillmem_relevance_threshold=0.4
skillmem_similarity_top_m=10
skillmem_similarity_backend=sentence_transformer
skillmem_model_name="${SKILLMEM_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
skillmem_counterfactual_mode=no_memory
skillmem_retrieval_query_mode=task_state
skillmem_state_aware_retrieval=true
skillmem_utility_update=baseline_relative
skillmem_step_utility_tau=1.0
skillmem_max_context_tokens=4096
skillmem_reflection_max_prompt_tokens=8192
skillmem_prompt_template_reserve_tokens=128

# Step-level and trajectory-level reflection for writing skills.
skillmem_step_reflection_enable=true
skillmem_step_reflection_per_group=1
skillmem_step_reflection_min_gap=0.05
reflection_per_group=1
reflection_selection=group_task
reflection_max_response_length=2048
reflection_temperature=1.0
reflection_top_p=1.0
reflection_backend="$UCOB_REFLECTION_BACKEND"

# Reflection self-training over written task/state memories.
reflection_train_enable=true
reflection_train_source=current_step_used_memory
reflection_train_warmup_steps=30
reflection_train_max_task_samples_per_step=16
reflection_train_max_state_samples_per_step=16
reflection_train_min_lifetime_count_task=1
reflection_train_min_lifetime_count_state=1
reflection_train_min_abs_utility=0.02
reflection_reward_mode=batch_zscore
reflection_advantage_group_by=memory_pool
reflection_advantage_eps=0.05
reflection_advantage_clip=2.0
reflection_train_mini_batch_size=16
reflection_train_response_length=$reflection_max_response_length
reflection_loss_weight=1.0

train_data_size=16
val_data_size=128
group_size=8

# Short run name; detailed settings are kept in this script.
experiment_name="ucob_full_alfworld_qwen3_1.7b"
skill_memory_file="outputs/dynamic_memory/${experiment_name}/alfworld_reflections.json"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --local_dir "${VERL_AGENT_DATA_DIR}" \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ucob \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=$gamma \
    algorithm.gigpo.step_advantage_w=$gigpo_step_advantage_w \
    algorithm.gigpo.mode=$gigpo_mode \
    algorithm.gigpo.enable_similarity=false \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.apply_chat_template_kwargs.enable_thinking=$require_thinking \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.max_model_len=$rollout_max_model_len \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=$rollout_top_p \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.credit_assignment=False \
    algorithm.filter_groups.enable=false \
    algorithm.dsdar.teacher_loss_enable=$teacher_loss_enable \
    algorithm.dsdar.sdar_coef=$legacy_teacher_coef \
    algorithm.dsdar.rollout_mode=$rollout_mode \
    algorithm.dsdar.adaptive_enable=$adaptive_enable \
    algorithm.dsdar.action_loss_mask_enable=$action_loss_mask_enable \
    algorithm.dsdar.topk_enable=$topk_enable \
    algorithm.ucob.cbsd.enable=$cbsd_enable \
    algorithm.ucob.cbsd.coef=$cbsd_coef \
    algorithm.ucob.cbsd.action_mask_enable=$cbsd_action_mask_enable \
    algorithm.ucob.cbsd.margin=$cbsd_margin \
    algorithm.ucob.cbsd.tau=$cbsd_tau \
    algorithm.ucob.cbsd.weight_max=$cbsd_weight_max \
    algorithm.ucob.cbsd.max_pairs_per_group=$cbsd_max_pairs_per_group \
    algorithm.ucob.cbsd.source_branches=$cbsd_source_branches \
    algorithm.ucob.cbsd.target_policy=$cbsd_target_policy \
    algorithm.ucob.cbsd.loss_type=$cbsd_loss_type \
    algorithm.ucob.cbsd.opd_granularity=$cbsd_opd_granularity \
    algorithm.ucob.cbsd.opd_scope=$cbsd_opd_scope \
    algorithm.ucob.cbsd.topk_k=$cbsd_topk_k \
    algorithm.ucob.cbsd.gate_enable=$cbsd_gate_enable \
    algorithm.ucob.cbsd.gate_beta=$cbsd_gate_beta \
    algorithm.ucob.cbsd.topk_norm_to_one=$cbsd_topk_norm_to_one \
    algorithm.ucob.cbsd.topk_clip_log_ratio=$cbsd_topk_clip_log_ratio \
    algorithm.ucob.cbsd.action_weight=$cbsd_action_weight \
    algorithm.ucob.cbsd.max_turns_per_trajectory=$cbsd_max_turns_per_trajectory \
    algorithm.ucob.skill_memory.enable=true \
    algorithm.ucob.skill_memory.filepath="$skill_memory_file" \
    algorithm.ucob.skill_memory.skill_format=claude_style \
    algorithm.ucob.skill_memory.category_filter=false \
    algorithm.ucob.skill_memory.skill_mapping_dir=skills/alfworld \
    algorithm.ucob.skill_memory.top_k=$skillmem_top_k \
    algorithm.ucob.skill_memory.eval_top_k=$skillmem_eval_top_k \
    algorithm.ucob.skill_memory.dual_pool_retrieval=$skillmem_dual_pool_retrieval \
    algorithm.ucob.skill_memory.memory_pool_mode=$skillmem_memory_pool_mode \
    algorithm.ucob.skill_memory.task_top_k=$skillmem_task_top_k \
    algorithm.ucob.skill_memory.state_top_k=$skillmem_state_top_k \
    algorithm.ucob.skill_memory.eval_task_top_k=$skillmem_eval_task_top_k \
    algorithm.ucob.skill_memory.eval_state_top_k=$skillmem_eval_state_top_k \
    algorithm.ucob.skill_memory.alpha=$skillmem_alpha \
    algorithm.ucob.skill_memory.beta=$skillmem_beta \
    algorithm.ucob.skill_memory.task_utility_beta=$skillmem_task_utility_beta \
    algorithm.ucob.skill_memory.state_utility_beta=$skillmem_state_utility_beta \
    algorithm.ucob.skill_memory.retrieve_type=ucb \
    algorithm.ucob.skill_memory.ucb_scale=$skillmem_ucb_scale \
    algorithm.ucob.skill_memory.filter_type=both \
    algorithm.ucob.skill_memory.write_enable=true \
    algorithm.ucob.skill_memory.validate_success=true \
    algorithm.ucob.skill_memory.utility_update=$skillmem_utility_update \
    algorithm.ucob.skill_memory.initial_utility_score=0.0 \
    algorithm.ucob.skill_memory.similarity_backend=$skillmem_similarity_backend \
    algorithm.ucob.skill_memory.model_name="$skillmem_model_name" \
    algorithm.ucob.skill_memory.relevance_threshold=$skillmem_relevance_threshold \
    algorithm.ucob.skill_memory.similarity_top_m=$skillmem_similarity_top_m \
    algorithm.ucob.skill_memory.counterfactual_mode=$skillmem_counterfactual_mode \
    algorithm.ucob.skill_memory.retrieval_query_mode=$skillmem_retrieval_query_mode \
    algorithm.ucob.skill_memory.state_aware_retrieval=$skillmem_state_aware_retrieval \
    algorithm.ucob.skill_memory.step_utility_tau=$skillmem_step_utility_tau \
    algorithm.ucob.skill_memory.max_context_tokens=$skillmem_max_context_tokens \
    algorithm.ucob.skill_memory.reflection_max_prompt_tokens=$skillmem_reflection_max_prompt_tokens \
    algorithm.ucob.skill_memory.prompt_template_reserve_tokens=$skillmem_prompt_template_reserve_tokens \
    algorithm.ucob.skill_memory.step_reflection_enable=$skillmem_step_reflection_enable \
    algorithm.ucob.skill_memory.step_reflection_per_group=$skillmem_step_reflection_per_group \
    algorithm.ucob.skill_memory.step_reflection_min_gap=$skillmem_step_reflection_min_gap \
    algorithm.ucob.skill_memory.merge_exact_task_attempt=false \
    algorithm.ucob.skill_memory.max_lessons_per_task_attempt=4 \
    algorithm.ucob.skill_memory.max_trajectory_steps=50 \
    algorithm.ucob.skill_memory.max_reflection_items=6 \
    algorithm.ucob.skill_memory.reflection_per_group=$reflection_per_group \
    algorithm.ucob.skill_memory.reflection_selection=$reflection_selection \
    algorithm.ucob.skill_memory.reflection_max_response_length=$reflection_max_response_length \
    algorithm.ucob.skill_memory.reflection_temperature=$reflection_temperature \
    algorithm.ucob.skill_memory.reflection_top_p=$reflection_top_p \
    algorithm.ucob.skill_memory.reflection_backend=$reflection_backend \
    algorithm.ucob.reflection.reflection_train_enable=$reflection_train_enable \
    algorithm.ucob.reflection.reflection_train_source=$reflection_train_source \
    algorithm.ucob.reflection.reflection_train_warmup_steps=$reflection_train_warmup_steps \
    algorithm.ucob.reflection.reflection_train_max_task_samples_per_step=$reflection_train_max_task_samples_per_step \
    algorithm.ucob.reflection.reflection_train_max_state_samples_per_step=$reflection_train_max_state_samples_per_step \
    algorithm.ucob.reflection.reflection_train_min_lifetime_count_task=$reflection_train_min_lifetime_count_task \
    algorithm.ucob.reflection.reflection_train_min_lifetime_count_state=$reflection_train_min_lifetime_count_state \
    algorithm.ucob.reflection.reflection_train_min_abs_utility=$reflection_train_min_abs_utility \
    algorithm.ucob.reflection.reflection_reward_mode=$reflection_reward_mode \
    algorithm.ucob.reflection.reflection_advantage_group_by=$reflection_advantage_group_by \
    algorithm.ucob.reflection.reflection_advantage_eps=$reflection_advantage_eps \
    algorithm.ucob.reflection.reflection_advantage_clip=$reflection_advantage_clip \
    algorithm.ucob.reflection.reflection_train_mini_batch_size=$reflection_train_mini_batch_size \
    algorithm.ucob.reflection.reflection_train_response_length=$reflection_train_response_length \
    algorithm.ucob.reflection.reflection_loss_weight=$reflection_loss_weight \
    env.env_name=alfworld/AlfredTWEnv \
    env.require_thinking=$require_thinking \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='UCOB_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.max_ckpt_to_keep=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    "$@"
