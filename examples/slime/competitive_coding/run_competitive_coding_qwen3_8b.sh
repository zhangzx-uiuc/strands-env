#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export project_name="slime_python_coding"
export exp_name="agentless_coding_v4"
export WANDB_API_KEY="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"

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

REPO_ROOT="/root/slime"
source "${REPO_ROOT}/scripts/models/qwen3-8B.sh"

TIMESTAMP_SUFFIX=$(date +%Y%m%d_%H%M%S)

CKPT_ARGS=(
   --hf-checkpoint /shared/models/qwen3_8b/hf
   --ref-load /shared/models/qwen3_8b/mcore
   --load /root/checkpoints/${project_name}/${exp_name}/
   --save /root/checkpoints/${project_name}/${exp_name}/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /shared/dev/zzx/data/slime_rl/source/Nemotron-RL-coding-competitive_coding/python_coding_slime_train_50.jsonl
   --rollout-max-prompt-len 2048
   --input-key messages
   --label-key label
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data valid /shared/dev/zzx/data/slime_rl/source/Nemotron-RL-coding-competitive_coding/python_coding_slime_valid_40.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 18432
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${project_name}
   --wandb-group ${exp_name}
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --dump-details /shared/dev/zzx/checkpoints/${project_name}/${exp_name}/details
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.slime.competitive_coding.generate_agentless.generate_and_rm
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${REPO_ROOT}/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
