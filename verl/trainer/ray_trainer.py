# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    ###########edit#################
    uid = data.non_tensor_batch["uid"]
    if isinstance(uid, np.ndarray) and uid.dtype == np.object_:
        try:
            uid = uid.astype(str).astype(np.int64)
        except:
            uid = np.array([hash(str(x)) for x in uid], dtype=np.int64)

    index_tensor = torch.tensor(uid, dtype=torch.long)
    ###########edit#################

    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    data.batch["advantage"] = advantages  # edit
    data.batch["index"] = index_tensor  # edit
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        print(f"run_time is {os.environ.get('Time', 'no_time')}")

        ###########edit FastRL #################
        # ========== f3 自适应 rollout.n (Adaptive Rollout Sampling) ==========
        # f3 = a * f1_mean + (1 - a) * f2_mean，第二个 epoch 起可用
        #   f1: 每个 step 内「每道题剪枝后平均保留条数」
        #   f2: 每个 step 内「每道题剪枝后最大保留条数」
        self.f3 = None
        self.g1_original = None  # 记录最原始的 f3（第一个 epoch 的统计量）
        self._f1_buffer = []  # 当前 epoch 内收集的所有 f1
        self._f2_buffer = []  # 当前 epoch 内收集的所有 f2
        self._last_epoch_id = 0  # 上次更新 f3 时所属的 epoch 编号
        self._n_original = config.worker.rollout.n  # 保存原始 rollout.n
        # ========== f3 自适应 rollout.n 结束 ==========

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        # when `update_steps` is set, the mini-batch size is derived from the actual batch size at each
        # update, so the rollout batch size does not need to be divisible by the actor global batch size.
        if (
            config.worker.actor.update_steps is None
            and config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0
        ):
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if config.worker.actor.update_steps is not None:
            # `update_steps` only requires the rollout batch to be evenly dispatched to DP ranks (see
            # `DataProto.chunk` and `_balance_batch`), which is guaranteed by this divisibility.
            dp_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
            if config.data.rollout_batch_size % dp_size != 0:
                raise ValueError(
                    "Rollout batch size must be divisible by the number of GPUs when `update_steps` is set, "
                    f"got rollout batch size {config.data.rollout_batch_size} and {dp_size} GPUs."
                )

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            # new_batch.non_tensor_batch["uid"] = np.array(
            #     [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            # )  #原来

            ###########edit#################
            # edit
            new_batch.non_tensor_batch["uid"] = np.array(
                [batch_dict["question_id"][i] for i in range(len(new_batch.batch))], dtype=np.int64
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            ###########edit FastRL #################
            # ========== 自适应 rollout 采样 (Adaptive Rollout Sampling) ==========
            # 依据上一个 epoch 统计出的 f3 与首个 epoch 的 g1_original 之差，
            # 动态收缩本 step 的 rollout.n；下界为 2（GRPO 分组至少需要 2 条）。
            rollout_sa = os.environ.get("RolloutSA", "false").lower() == "true"
            if rollout_sa:
                print("启动动态采样！")
                if self.f3 is not None and self.g1_original is not None and self.f3 < self.g1_original:
                    n_new = self._n_original - (self.g1_original - self.f3)
                    self.config.worker.rollout.n = max(int(round(min(n_new, self._n_original))), 2)
                else:
                    self.config.worker.rollout.n = self._n_original
            else:
                self.config.worker.rollout.n = self._n_original
                print("不启动动态采样！")

            print(
                f"rollout.n 为：{self.config.worker.rollout.n}  "
                f"(原始={self._n_original}, g1={self.g1_original}, f3={self.f3})"
            )
            gen_batch.meta_info["n"] = self.config.worker.rollout.n  # edit
            ###########edit FastRL #################

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                ###########edit FastRL #################
                # FastRL 相关开关（统一在此读取，供下面的重排/剪枝分支使用）
                run_time = os.environ.get("Time", "no_time")
                a_pruning = os.getenv("PRUNING", "false").lower() == "true"
                sim_threshold = float(os.getenv("SIM_THRESHOLD", "0.2"))
                baseline_cppo = os.getenv("CPPO", "false").lower() == "true"

                # ====================================================================
                # FastRL ordering optimization:
                # 原顺序: old_log_probs -> ref -> values -> adv -> pruning -> update
                # 新顺序: adv -> pruning -> old_log_probs -> ref -> values -> update
                # 理由: 剪枝会显著缩小 batch，把 old/ref/values 三个前向放到剪枝之后
                #       可直接省下 timing_s/old 与 timing_s/ref 的墙钟时间。
                # 安全前置条件:
                #   - apply_kl_penalty 需要 old_log_probs + ref_log_probs（见 apply_kl_penalty），
                #     因此只有在 `use_kl_loss=True` 或没有 reference policy 时才能重排；
                #   - GAE 需要 critic 的 values 才能算 advantage，因此 use_critic=True 时不能重排；
                #   - CPPO 分支需要 old_log_probs 计算策略熵，开启时不重排；
                #   - 不剪枝时重排没有收益，保持原顺序以保证与 baseline 数值完全一致。
                # 任一条件不满足则回退到原始顺序。
                # ====================================================================
                _can_reorder = (
                    (self.config.algorithm.use_kl_loss or not self.use_reference_policy)
                    and not self.use_critic
                    and a_pruning
                    and not baseline_cppo
                )

                # ----- 回退路径: 重排不安全时保持原始顺序 -----
                if not _can_reorder:
                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                ###########edit FastRL #################

                # ----- adv 块（总是执行）。_can_reorder=True 时它跑在
                #       old/ref/values 之前，从而让剪枝先于三个前向发生。-----
                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # CPPO  https://arxiv.org/abs/2503.22342
                ###########################  index  #############################################
                # # 记录日志
                if baseline_cppo:
                    print("启动CPPO！")
                    txt_lines = []
                    pruning_rate = 0.5
                    retain_rate = 1 - pruning_rate
                    txt_lines.append(f"Pruning 前 advantages.shape: {batch.batch['advantage'].shape}")

                    advantages = batch.batch['advantage']  # [B*n, seq_len]
                    batch_indexes = batch.batch["index"].cpu().numpy()  # 每个样本的题目 ID

                    mask = batch.batch["response_mask"].float()  # 转成 float
                    # 计算每条样本的平均 advantage绝对值（只考虑有效 token）
                    avg_advantages = ((advantages * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)).abs()
                    avg_advantages_1 = ((advantages * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8))

                    old_log_probs = batch.batch["old_log_probs"]  # [B*n, seq_len]
                    probs = old_log_probs.exp()  # log_probs -> probs
                    sample_entropy = -(probs * old_log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

                    # n, B, k
                    n = self.config.worker.rollout.n
                    B = len(batch) // n
                    k = max(1, int(n * (retain_rate)))

                    txt_lines.append(f"原始总题数 B: {B}, 每题候选数 n: {n}, 保留数 k: {k}")

                    # ========== ① 按 index 分组 ==========
                    index_to_indices = defaultdict(list)

                    for global_idx, qid in enumerate(batch_indexes):
                        index_to_indices[qid].append(global_idx)

                    # 日志：统计每个题目的 rollout 数
                    txt_lines.append("\n===== Pruning 前：每个 index（题目）的样本数 =====")
                    for qid in sorted(index_to_indices.keys()):
                        txt_lines.append(f"index={qid}: {len(index_to_indices[qid])} 个样本")

                    # 验证
                    txt_lines.append(f"\n总样本数（应等于 B*n）: {sum(len(v) for v in index_to_indices.values())}\n")

                    # ========== ② 对每个 index（每道题）做 top-k 裁剪 ==========
                    retained_indices = []

                    for qid, sample_indices in index_to_indices.items():
                        sample_adv = avg_advantages[sample_indices]  # 这道题的 n 个 advantage
                        sample_adv_1 = avg_advantages_1[sample_indices]
                        sample_ent = sample_entropy[sample_indices]  # 对应的策略熵

                        # 排序取 top-k
                        topk = torch.topk(sample_adv, k=k, largest=True)
                        topk_local_idx = topk.indices.cpu().numpy()  # 在本题组内的局部索引
                        topk_global = [sample_indices[j] for j in topk_local_idx]

                        # 记录日志
                        txt_lines.append(f"===== 题目 index={qid} =====")
                        txt_lines.append(f"所有 {len(sample_indices)} 个 abs_advantage: {sample_adv.cpu().numpy().round(4)}")
                        txt_lines.append(f"所有 {len(sample_indices)} 个 advantage: {sample_adv_1.cpu().numpy().round(4)}")
                        txt_lines.append(f"所有 {len(sample_indices)} 个策略熵: {sample_ent.cpu().numpy().round(4)}")
                        txt_lines.append(f"保留的 {k} 个 advantage: {topk.values.cpu().numpy().round(4)}")
                        txt_lines.append(f"保留全局 idx: {topk_global}\n")

                        retained_indices.extend(topk_global)

                    # ========== ③ 合并并排序 ==========
                    retained_indices = torch.tensor(retained_indices).sort()[0]

                    # 更新 batch
                    batch = batch[retained_indices]

                    txt_lines.append("===== Pruning 后全局信息 =====")
                    txt_lines.append(f"Pruning 后总样本数: {len(retained_indices)}")
                    txt_lines.append(f"Pruning 后 advantages.shape: {batch.batch['advantages'].shape}")

                    # 保存日志
                    cppo_log_path = os.path.join(
                        os.getenv("PRUNING_LOG_DIR", "logs/geo3k"), f"pruning_cppo_grpo_{run_time}.txt"
                    )
                    os.makedirs(os.path.dirname(cppo_log_path), exist_ok=True)
                    with open(cppo_log_path, "a") as f:
                        for line in txt_lines:
                            f.write(line + "\n")

                    print(f"Pruning 日志已保存到 {cppo_log_path}")
                else:
                    print("不启动CPPO！")

                ###########edit FastRL #################
                ###########################################################################################################
                # 最大化差异剪枝 FastRL (ours)
                # Phase 1: 组内按 advantage 去重 + Token Jaccard 相似度保护（低相似度=不同推理路径，保留）
                # Phase 2: round-robin 按 |advantage| 回补，使 batch 能被 divisor 整除
                #   - 未设置 update_steps: divisor = global_batch_size * n（保证 mini-batch 均分）
                #   - 设置 update_steps: divisor = world_size（更新次数固定，只需能被 DP 均分）
                ###########################################################################################################
                if a_pruning:
                    print("启动最大化差异剪枝！！！")
                    txt_lines = []

                    txt_lines.append("\n===== Pruning 前全局信息 =====")
                    txt_lines.append(f"总样本数: {len(batch)}")
                    txt_lines.append(f"advantages.shape: {batch.batch['advantage'].shape}")

                    advantages = batch.batch["advantage"]
                    batch_indexes = batch.batch["index"].cpu().numpy()
                    responses = batch.batch["responses"]
                    mask = batch.batch["response_mask"].float()

                    avg_advantages = (advantages * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    avg_advantages_abs = avg_advantages.abs()

                    n = self.config.worker.rollout.n
                    global_batch_size = self.config.worker.actor.global_batch_size
                    update_steps = self.config.worker.actor.update_steps
                    if update_steps is not None:
                        # policy 更新次数固定为 update_steps，mini-batch 大小按实际样本数动态切分，
                        # 因此只需保证剪枝后的 batch 能被 DP（GPU 数）整除。
                        divisor = self.actor_rollout_ref_wg.world_size
                    else:
                        divisor = global_batch_size * self._n_original

                    # ---------- ① 按 index 分组 ----------
                    index_to_indices = defaultdict(list)
                    for global_idx, qid in enumerate(batch_indexes):
                        index_to_indices[qid].append(global_idx)

                    sorted_qids = sorted(index_to_indices.keys())

                    selected_per_qid = defaultdict(list)
                    unselected_per_qid = defaultdict(list)

                    # =========================================================
                    # Phase 1：最大化差异 + 相似度保护
                    # =========================================================

                    def _token_jaccard(resp_a, mask_a, resp_b, mask_b):
                        """Compute Token Jaccard similarity between two responses."""
                        len_a = int(mask_a.sum().item())
                        len_b = int(mask_b.sum().item())
                        set_a = set(resp_a[:len_a].cpu().tolist())
                        set_b = set(resp_b[:len_b].cpu().tolist())
                        inter = len(set_a & set_b)
                        union = len(set_a | set_b)
                        return inter / union if union > 0 else 1.0

                    for qid in sorted_qids:
                        sample_indices = index_to_indices[qid]

                        raw = avg_advantages[sample_indices].cpu().numpy()
                        abs_val = avg_advantages_abs[sample_indices].cpu().numpy()
                        raw_round4 = raw.round(4)

                        value_to_locals = defaultdict(list)
                        for i, v in enumerate(raw_round4):
                            value_to_locals[v].append(i)

                        selected_local = set()

                        txt_lines.append(f"\n===== qid={qid} | n_responses={len(sample_indices)} =====")
                        txt_lines.append(f"  avg_advantages (raw):    {raw.round(6)}")
                        txt_lines.append(f"  avg_advantages (round4): {raw_round4}")
                        txt_lines.append(f"  abs_advantages:          {abs_val.round(6)}")
                        txt_lines.append(f"  unique advantage groups: {len(value_to_locals)}")

                        for v, locals_ in value_to_locals.items():
                            # Always keep the one with largest abs advantage as seed
                            best_local = locals_[abs_val[locals_].argmax()]
                            selected_local.add(best_local)

                            txt_lines.append(f"  --- adv_group={v} | count={len(locals_)} ---")
                            txt_lines.append(
                                f"      seed: local_idx={best_local}, global_idx={sample_indices[best_local]}, "
                                f"abs_adv={abs_val[best_local]:.6f} -> KEEP(seed)"
                            )

                            # For other responses with same advantage, check similarity with seed
                            if len(locals_) > 1:
                                seed_gidx = sample_indices[best_local]
                                seed_resp = responses[seed_gidx]
                                seed_mask = mask[seed_gidx]
                                seed_valid_len = int(seed_mask.sum().item())

                                for loc in locals_:
                                    if loc == best_local:
                                        continue
                                    cand_gidx = sample_indices[loc]
                                    sim = _token_jaccard(seed_resp, seed_mask, responses[cand_gidx], mask[cand_gidx])
                                    cand_valid_len = int(mask[cand_gidx].sum().item())
                                    # Low similarity means different reasoning path, keep it
                                    if sim < sim_threshold:
                                        selected_local.add(loc)
                                        decision = "KEEP(low_sim)"
                                    else:
                                        decision = "PRUNE(high_sim)"
                                    txt_lines.append(
                                        f"      cand:  local_idx={loc}, global_idx={cand_gidx}, "
                                        f"abs_adv={abs_val[loc]:.6f}, jaccard_sim={sim:.4f} "
                                        f"(threshold={sim_threshold}), seed_len={seed_valid_len}, "
                                        f"cand_len={cand_valid_len} -> {decision}"
                                    )

                        txt_lines.append(
                            f"  >> selected: {sorted(selected_local)} ({len(selected_local)}/{len(sample_indices)})"
                        )

                        for i, gidx in enumerate(sample_indices):
                            if i in selected_local:
                                selected_per_qid[qid].append(gidx)
                            else:
                                unselected_per_qid[qid].append(gidx)

                    # ---------- Phase1 合并 ----------
                    retained_indices = []
                    for qid in sorted_qids:
                        retained_indices.extend(selected_per_qid[qid])

                    total_retained = len(retained_indices)

                    # =========================================================
                    # Phase 2：round-robin 补齐
                    # =========================================================
                    need = (divisor - total_retained % divisor) % divisor
                    added_per_qid = defaultdict(list)

                    if need > 0:
                        for qid in sorted_qids:
                            unselected_per_qid[qid] = sorted(
                                unselected_per_qid[qid],
                                key=lambda i: avg_advantages_abs[i].item(),
                                reverse=True,
                            )

                        added = []
                        ptr = 0
                        qnum = len(sorted_qids)

                        while need > 0:
                            qid = sorted_qids[ptr % qnum]

                            if unselected_per_qid[qid]:
                                idx = unselected_per_qid[qid].pop(0)
                                added.append(idx)
                                added_per_qid[qid].append(idx)
                                need -= 1

                            ptr += 1
                            if ptr > qnum * 2 and not any(unselected_per_qid[q] for q in sorted_qids):
                                break

                        retained_indices.extend(added)

                    # =========================================================
                    # 最终整理 batch
                    # =========================================================
                    retained_indices = torch.tensor(sorted(retained_indices), device=avg_advantages.device)

                    batch = batch[retained_indices]

                    # =========================================================
                    # 统计 f1 / f2 / f3
                    # =========================================================
                    final_counts = []

                    for qid in sorted_qids:
                        phase1_count = len(selected_per_qid[qid])
                        phase2_count = len(added_per_qid[qid])
                        final_counts.append(phase1_count + phase2_count)

                    if final_counts:
                        f1 = sum(final_counts) / len(final_counts)
                        f2 = max(final_counts)
                    else:
                        f1 = 0
                        f2 = 0

                    # =========================================================
                    # 累积 f1/f2，按 epoch 边界更新 f3
                    # f3 = a * f1_mean + (1-a) * f2_mean
                    # 其中 a = global_step / training_steps（随训练递增）
                    # =========================================================
                    self._f1_buffer.append(f1)
                    self._f2_buffer.append(f2)

                    steps_per_epoch = self.training_steps / max(self.config.trainer.total_epochs, 1)
                    current_epoch_id = int((self.global_step - 1) // steps_per_epoch)  # 0-based

                    if current_epoch_id > self._last_epoch_id:
                        # 跨入了新 epoch，用上一个 epoch 收集的 f1/f2 加权更新 f3
                        f1_mean, f2_mean = None, None
                        if self._f1_buffer and self._f2_buffer:
                            f1_mean = sum(self._f1_buffer) / len(self._f1_buffer)
                            f2_mean = sum(self._f2_buffer) / len(self._f2_buffer)
                            a = self.global_step / self.training_steps
                            self.f3 = a * f1_mean + (1 - a) * f2_mean

                        self._f1_buffer = []  # 重置缓冲区
                        self._f2_buffer = []  # 重置缓冲区
                        self._last_epoch_id = current_epoch_id
                        if self._last_epoch_id == 1 and f1_mean is not None:
                            self.g1_original = (f1_mean + f2_mean) / 2

                        f3_update_msg = (
                            f"[f3 更新] epoch {current_epoch_id}, f1_mean= {f1_mean}, f2_mean = {f2_mean}, "
                            f"f3 = {self.f3}, g1_original = {self.g1_original}"
                        )
                        print(f3_update_msg)
                        txt_lines.append(f3_update_msg)

                    txt_lines.append("\n" + "=" * 60)
                    txt_lines.append("===== 整个 Batch 剪枝后统计 =====")
                    txt_lines.append(f"每个 question_id 最终保留条数 = {final_counts}")
                    txt_lines.append(f"整个 Batch  f1 (平均保留数) = {f1:.4f}")
                    txt_lines.append(f"整个 Batch  f2 (最大保留数) = {f2}")

                    txt_lines.append("\n===== Pruning 后 =====")
                    txt_lines.append(f"本 step rollout.n = {n} (原始 = {self._n_original})")
                    txt_lines.append(f"最终样本数: {len(retained_indices)}")
                    txt_lines.append(
                        f"对齐 divisor = {divisor} (update_steps = {update_steps}), "
                        f"是否整除: {len(retained_indices) % divisor == 0}"
                    )

                    # ---------- 保存日志 ----------
                    log_path = os.path.join(
                        os.getenv("PRUNING_LOG_DIR", "logs/geo3k"),
                        f"pruning_debug_fastrl_{global_batch_size}_{run_time}.txt",
                    )
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    with open(log_path, "a") as f:
                        for line in txt_lines:
                            f.write(line + "\n")

                    print(f"Pruning 日志已保存到 {log_path}")

                else:
                    print("不启动最大化差异剪枝！！！")

                # ----- 剪枝后重新做 DP token 均衡 -----
                # 剪枝会按 advantage/相似度任意删样本，破坏前面 _balance_batch 建立的
                # 各 dp rank token 均衡；且 meta_info["global_token_num"] 仍是剪枝前的
                # 长度，会让 worker 端 MFU / 吞吐统计失真。这里重建两者。
                # _balance_batch 只做重排序，幂等且无副作用（old/ref/advantages 均按样本对齐）。
                if a_pruning or baseline_cppo:
                    self._balance_batch(batch, metrics=metrics, logging_prefix="global_seqlen_after_pruning")
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # ----- 重排路径: 在（可能已剪枝的）小 batch 上算三个概率分布 -----
                if _can_reorder:
                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                ###########edit FastRL #################

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
