#!/bin/bash
# A.2 smoke — sandbox round-robin across N2D MIG pool (4 workers).
# r2egym's docker.from_env() is monkey-patched at import time by
# examples/deepswe/mig_docker_balancer.py to pick a worker per RepoEnv.
# See tasks/tinyflow_integration/track_a.md A.2.
#
# Pre-req: MIG_WORKER_IPS env var = comma-separated worker internal IPs.

set -euo pipefail

export SKIP_JAX_PRECOMPILE=true
# DO NOT export DOCKER_HOST here — the patcher overrides per call.
# (Setting it would force ALL calls to one host, defeating round-robin.)
unset DOCKER_HOST

if [[ -z "${MIG_WORKER_IPS:-}" ]]; then
  echo "ERROR: MIG_WORKER_IPS env var is required (comma-separated worker IPs)." >&2
  exit 2
fi

# ── Model: Qwen3-4B (smaller than v5p-32 reference's Qwen3-32B) ─────────────
model_name="${model_name:-Qwen3-4B-Instruct-2507}"
model_id="${model_id:-Qwen/Qwen3-4B-Instruct-2507}"
tokenizer_path="${tokenizer_path:-$model_id}"

# ── Training loop (smoke = 2 batches, 1 epoch) ──────────────────────────────
num_batches="${num_batches:-1}"
num_train_epochs="${num_train_epochs:-1}"
train_fraction="${train_fraction:-1.0}"
warmup_ratio="${warmup_ratio:-0.1}"

batch_size="${batch_size:-1}"
mini_batch_size="${mini_batch_size:-1}"
train_micro_batch_size="${train_micro_batch_size:-1}"
rollout_micro_batch_size="${rollout_micro_batch_size:-1}"

num_generations="${num_generations:-2}"
max_response_length="${max_response_length:-2048}"

# ── Mesh: 4 chips colocated — trainer + rollout share same actor mesh ──────
trainer_mesh="${trainer_mesh:-(2,2)}"
# rollout/reference do not own a mesh; they reuse actor's via same_mesh_as below

# ── Local checkpoint dir (no GCS for smoke) ─────────────────────────────────
checkpoint_dir="${checkpoint_dir:-/mnt/disks/tunix-data/checkpoints/smoke_a2}"
checkpoint_suffix="${checkpoint_suffix:-$(printf '%04d' "$((RANDOM % 10000))")}"
if [[ -n "$checkpoint_dir" && "$checkpoint_dir" != "null" ]]; then
  checkpoint_dir="${checkpoint_dir}_${checkpoint_suffix}"
fi

max_steps=$(awk "BEGIN {
  value = $num_batches * $num_train_epochs * $train_fraction;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")
warmup_steps=$(awk "BEGIN {
  value = $warmup_ratio * $max_steps;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")
vllm_max_num_seqs=$(awk "BEGIN {
  value = $rollout_micro_batch_size * $num_generations;
  if (value < 1) value = 1;
  printf \"%.0f\", value;
}")

python -c '
import sys, faulthandler
faulthandler.enable()
# Dump every thread stack to stderr after 60s (and every 60s thereafter) — for A.2 hang diagnostics
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
sys.path.insert(0, "examples/deepswe")
import mig_docker_balancer  # round-robin docker.from_env() across MIG workers
sys.argv[0] = "tunix.cli.grpo_main"
from absl import app
from tunix.cli.grpo_main import main
app.run(main)
' \
  tunix/cli/base_agentic_config.yaml \
  \
  `# ── Model ────────────────────────────────────────────────────────────` \
  model_config.model_name="$model_name" \
  model_config.model_id="$model_id" \
  model_config.model_source="huggingface" \
  model_config.rng_seed=42 \
  model_config.model_display=false \
  model_config.remat_config=3 \
  actor_model_config.mesh.shape="$trainer_mesh" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  reference_model_config.mesh=null \
  reference_model_config.same_mesh_as="actor" \
  rollout_model_config.mesh=null \
  rollout_model_config.same_mesh_as="actor" \
  \
  `# ── Data: take only first 4 instances of R2E-Gym-V1 train split ─────` \
  data_module="examples.deepswe.deepswe_data" \
  data_config.dataset_name="R2E-Gym/R2E-Gym-V1" \
  data_config.dataset_split="train[:4]" \
  data_config.shuffle=true \
  data_config.seed=42 \
  prompt_key="problem_statement" \
  \
  `# ── Training loop (agentic GRPO, smoke = 2 batches) ─────────────────` \
  training_mode="agentic_grpo" \
  batch_size="$batch_size" \
  num_batches="$num_batches" \
  num_train_epochs="$num_train_epochs" \
  train_fraction="$train_fraction" \
  reward_functions=[] \
  verl_compatible=false \
  \
  `# ── Rollout engine: vLLM in-process ──────────────────────────────────` \
  rollout_engine="vllm" \
  offload_to_cpu=false \
  \
  `# ── Rollout config (response cap reduced for smoke) ─────────────────` \
  rollout_config.max_prompt_length=4096 \
  rollout_config.total_generation_steps="$max_response_length" \
  rollout_config.max_tokens_to_generate="$max_response_length" \
  rollout_config.temperature=1.0 \
  rollout_config.top_p=null \
  rollout_config.top_k=null \
  rollout_config.return_logprobs=true \
  \
  `# ── vLLM config ──────────────────────────────────────────────────────` \
  vllm_config.hbm_utilization=0.4 \
  vllm_config.tpu_backend_type="jax" \
  vllm_config.server_mode=true \
  vllm_config.async_scheduling=true \
  vllm_config.max_num_seqs="$vllm_max_num_seqs" \
  vllm_config.kwargs.kv_cache_metrics=true \
  vllm_config.kwargs.disable_log_stats=false \
  vllm_config.kwargs.enable_prefix_caching=true \
  \
  `# ── Chat / agent / env wiring ────────────────────────────────────────` \
  chat_parser_config.type="qwen" \
  tokenizer_config.tokenizer_type="huggingface" \
  tokenizer_config.tokenizer_path="$tokenizer_path" \
  tokenizer_config.add_bos=false \
  tokenizer_config.add_eos=false \
  agent_class_path="examples.deepswe.swe_agent.SWEAgent" \
  env_class_path="examples.deepswe.swe_env.SWEEnv" \
  env_kwargs.max_steps=4 \
  env_kwargs.backend=docker \
  \
  `# ── Agentic / multi-turn (smoke = max 4 turns) ───────────────────────` \
  agentic_grpo_config.max_turns=4 \
  agentic_grpo_config.per_turn_timeout_secs=120 \
  agentic_grpo_config.context_ratio=2 \
  agentic_grpo_config.max_concurrency=4 \
  \
  `# ── GRPO algorithm ───────────────────────────────────────────────────` \
  agentic_grpo_config.num_generations="$num_generations" \
  agentic_grpo_config.max_response_length="$max_response_length" \
  agentic_grpo_config.num_iterations=1 \
  agentic_grpo_config.beta=0.001 \
  agentic_grpo_config.epsilon=0.2 \
  agentic_grpo_config.epsilon_high=0.28 \
  agentic_grpo_config.off_policy_steps=0 \
  agentic_grpo_config.loss_agg_mode="sequence-mean-token-mean" \
  agentic_grpo_config.kl_loss_mode="low_var_kl" \
  \
  `# ── Optimizer ────────────────────────────────────────────────────────` \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.learning_rate=1e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=1e-6 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio="$warmup_ratio" \
  rl_training_config.actor_optimizer_config.warmup_steps="$warmup_steps" \
  rl_training_config.actor_optimizer_config.decay_steps="$max_steps" \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  \
  `# ── RL training ──────────────────────────────────────────────────────` \
  rl_training_config.eval_every_n_steps=10 \
  rl_training_config.max_steps="$max_steps" \
  rl_training_config.mini_batch_size=1 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.rollout_micro_batch_size=1 \
  rl_training_config.compute_logps_micro_batch_size=1 \
  rl_training_config.checkpoint_root_directory="$checkpoint_dir" \
  rl_training_config.checkpointing_options.save_interval_steps=100 \
  rl_training_config.checkpointing_options.max_to_keep=1 \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/deepswe_smoke_a2" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=2 \
  \
  "$@"
