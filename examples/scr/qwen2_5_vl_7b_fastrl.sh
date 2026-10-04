#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_DEEP_GEMM=0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export Time=$(date +"%Y%m%d_%H%M%S")
export RolloutSA=false  #需要开启则开
export PRUNING=true     #fastrl核心
export CPPO=false
export SIM_THRESHOLD=0.2
export PRUNING_LOG_DIR=logs/geo3k
export RAY_process_group_cleanup_enabled=true
# 获取当前时间（格式：YYYYMMDD_HHMMSS）
# RolloutSA=true  启动自适应 rollout 采样（动态收缩 rollout.n）
# PRUNING=true    启动最大化差异的 advantage 剪枝
# SIM_THRESHOLD   同 advantage 组内的 Token Jaccard 相似度阈值，低于该值视为不同推理路径并保留
# worker.actor.global_batch_size=16即mini_batch_size=16，为fastrl与baseline的policy_model更新次数保持一致（因为剪枝后轨迹变少，更新次数也会变少），因此将mini_batch_size设置的更小，
# 我们后续将对两者的update次数（例如4）作为固定参数将两者保持一致，就不需要额外设置mini_batch_size了

MODEL_PATH=pretrain_model/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config_geo3k.yaml \
    data.train_files=data/geometry3k/train-00000-of-00001.parquet \
    data.val_files=data/geometry3k/test-00000-of-00001.parquet \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=16 \
    trainer.experiment_name=Qwen2.5-VL-7B-Instruct-FastRL \
    trainer.save_checkpoint_path=Save_model/geo3k/GRPO/Qwen2.5-VL-7B-Instruct-FastRL \
    trainer.total_epochs=25 \
    trainer.n_gpus_per_node=4
