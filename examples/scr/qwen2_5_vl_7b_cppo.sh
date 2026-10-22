#!/bin/bash

set -x
export VLLM_USE_DEEP_GEMM=0
export Time=$(date +"%Y%m%d_%H%M%S")
export RolloutSA=false
export PRUNING=false
export CPPO=true
export SIM_THRESHOLD=0.2
export RAY_process_group_cleanup_enabled=true
# 获取当前时间（格式：YYYYMMDD_HHMMSS）
# 启动rollout优化
# 启动剪枝

MODEL_PATH=pretrain_model/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_geo3k.yaml \
    data.train_files=data/geometry3k/train-00000-of-00001.parquet \
    data.val_files=data/geometry3k/test-00000-of-00001.parquet \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.update_steps=4 \
    trainer.experiment_name=Qwen2.5-VL-7B-Instruct-CPPO \
    trainer.save_checkpoint_path=Save_model/geo3k/GRPO/Qwen2.5-VL-7B-Instruct-CPPO \
    trainer.total_epochs=25 \
    trainer.n_gpus_per_node=4
