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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

WorkerType = Type[Worker]

ZERO_TEMP = "A conversation between User and Assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The final answer is concluded in the format of \\boxed{{}}.\n\nUser: {prompt}.\nAssistant: Let me solve this step by step."


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Proposer = 7
    ProposerRollout = 8
    ProposerRolloutRef = 9
    ProposerRefPolicy = 10


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "reinforce_plus_plus":
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                .detach()
                .item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(
            torch.eq(prompt_length, max_prompt_length).float()
        )
        .detach()
        .item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name]
            * 1000
            / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef
                )
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "reinforce_plus_plus":
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(
                    f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                    f"'{name}.micro_batch_size_per_gpu'."
                )

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.micro_batch_size' AND "
                    f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                    f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)
                > 1
            ):
                assert (
                    config.actor_rollout_ref.model.use_remove_padding
                ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert (
                    config.critic.model.use_remove_padding
                ), "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and "wandb" not in self.config.trainer.logger:
            print(
                "WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. "
            )
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"]
                for i in range(len(samples))
            ],
            [],
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"]
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._maybe_log_val_generations_to_wandb(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        import dill

        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if not (
                self.config.trainer.resume_from_path and global_step_folder is not None
            ):
                assert isinstance(
                    self.config.trainer.resume_mode, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_mode
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)
        return global_idx

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"]
                )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get(
                            "use_kl_loss", False
                        ):
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return


class RayCPTrainer(RayPPOTrainer):

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        proposer_tokenizer=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        if proposer_tokenizer is None:
            proposer_tokenizer = tokenizer
        self.proposer_tokenizer = proposer_tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.solver_zero = config.actor_rollout_ref.actor.get("solver_zero", False)
        self.proposer_ema = config.actor_rollout_ref.actor.get("proposer_ema", False)
        self.proposer_ema_decay = config.actor_rollout_ref.actor.get(
            "proposer_ema_decay", 0.99
        )

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef
                )
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "reinforce_plus_plus":
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        self.val_score_history = []
        self.actor_val_score_history = []

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

        # TODO: we have to make sure the batch size is divisible by the dp size

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
            zero=self.solver_zero,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.val_dataloader) >= 1

        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(
            Role.ProposerRollout
        )
        proposer_actor_config = deepcopy(self.config.actor_rollout_ref)
        proposer_actor_config.model.path = proposer_actor_config.model.proposer_path
        proposer_actor_config.rollout.n = 1
        proposer_actor_config.actor.optim.lr = proposer_actor_config.actor.optim.get(
            "proposer_lr", proposer_actor_config.actor.optim.lr
        )
        proposer_actor_config.actor.ppo_mini_batch_size = (
            proposer_actor_config.actor.ppo_mini_batch_size
            * self.config.trainer.proposer_update_interval
        )
        proposer_actor_config.rollout.temperature = proposer_actor_config.rollout.get(
            "proposer_temperature", proposer_actor_config.rollout.temperature
        )
        proposer_actor_config.rollout.top_p = proposer_actor_config.rollout.get(
            "proposer_top_p", proposer_actor_config.rollout.top_p
        )
        proposer_actor_config.actor.entropy_coeff = proposer_actor_config.actor.get(
            "proposer_entropy_coeff", proposer_actor_config.actor.entropy_coeff
        )
        proposer_actor_config.actor.kl_loss_coef = proposer_actor_config.actor.get(
            "proposer_kl_loss_coef", proposer_actor_config.actor.kl_loss_coef
        )
        proposer_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ProposerRollout],
            config=proposer_actor_config,
            role="proposer_rollout",
        )
        self.resource_pool_to_cls[resource_pool][
            "proposer_rollout"
        ] = proposer_rollout_cls

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ProposerRefPolicy
            )
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.ProposerRefPolicy],
                config=proposer_actor_config,
                role="proposer_ref",
            )
            self.resource_pool_to_cls[resource_pool]["proposer_ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        self.proposer_rollout_wg = all_wg["proposer_rollout"]
        self.proposer_rollout_wg.init_model()

        self.proposer_ref_wg = all_wg["proposer_ref"]
        self.proposer_ref_wg.init_model()

    def _validate(self):
        
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"]
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.actor.get(
                    "val_do_sample", False
                ),
                "validate": True,
            }
            
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            if self.config.actor_rollout_ref.actor.get(
                "proposer_train_metric", "train_reward"
            ) == "val_reward" and self.config.actor_rollout_ref.actor.get(
                "val_use_rm", True
            ):
                reward_tensor = self.rm_wg.compute_rm_score(test_batch)
                test_batch = test_batch.union(reward_tensor)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )
        
        self._maybe_log_val_generations_to_wandb(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict

    def _generate_instructions(self, batch_size: int):
        print("Generating instructions")
        prompt_with_chat_template = self.proposer_tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": self.config.data.get(
                        "proposer_prompt", "Generate a valid math problem."
                    ),
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        
        length = len(self.proposer_tokenizer.tokenize(prompt_with_chat_template))
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.proposer_tokenizer,
            max_length=length,
            pad_token_id=self.proposer_tokenizer.pad_token_id,
            left_pad=True,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        data = {
            "input_ids": input_ids.repeat(batch_size, 1),
            "attention_mask": attention_mask.repeat(batch_size, 1),
            "position_ids": position_ids.repeat(batch_size, 1),
        }
        proposer_batch = DataProto.from_single_dict(data)
        gen_batch = proposer_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"]
        )

        instructions = self.proposer_rollout_wg.generate_sequences(gen_batch)
        proposer_batch = proposer_batch.union(instructions)
        instructions_text = self.proposer_tokenizer.batch_decode(
            instructions.batch["responses"], skip_special_tokens=True
        )
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        for text in instructions_text:
            if self.solver_zero:
                prompt_with_chat_template = ZERO_TEMP.format(prompt=text)
            else:
                prompt_with_chat_template = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.actor_rollout_ref.get("truncation", "right"),
            )
            position_ids = compute_position_id_with_mask(attention_mask[0])

            input_ids_list.append(input_ids[0])
            attention_mask_list.append(attention_mask[0])
            position_ids_list.append(position_ids)

        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        position_ids = torch.stack(position_ids_list, dim=0)

        raw_prompt = np.array(
            [[{"role": "user", "content": text}] for text in instructions_text]
        )
        data = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "raw_prompt": raw_prompt,
            }
        )
        return data, proposer_batch

    def _proposer_train_reward(self, batch: DataProto, group_indexes: dict):
        new_batch = {}
        for _, indexes in group_indexes.items():
            for key, value in batch.batch.items():
                if key not in new_batch:
                    new_batch[key] = []
                if key != "rm_scores":
                    new_batch[key].append(value[indexes[0]])
                else:
                    rewards = value[indexes]
                    rewards_index = torch.nonzero(rewards, as_tuple=True)
                    rewards_value = rewards[rewards_index]
                    reward_type = self.config.actor_rollout_ref.actor.get(
                        "proposer_reward_type", "mean/std"
                    )
                    reward_up_bound = self.config.actor_rollout_ref.actor.get(
                        "proposer_reward_up_bound", None
                    )
                    reward_low_bound = self.config.actor_rollout_ref.actor.get(
                        "proposer_reward_low_bound", None
                    )
                    if reward_type == "mean/std":
                        final_reward = -torch.mean(rewards_value) / (
                            torch.std(rewards_value) + 1e-4
                        )
                    elif reward_type == "mean*std":
                        final_reward = -torch.mean(rewards_value) * torch.std(
                            rewards_value
                        )
                    else:
                        raise NotImplementedError
                    if reward_up_bound is not None:
                        final_reward = torch.clamp(final_reward, max=reward_up_bound)
                    if reward_low_bound is not None:
                        final_reward = torch.clamp(final_reward, min=reward_low_bound)

                    length = batch.batch["responses"].shape[1]
                    real_length = torch.sum(
                        batch.batch["responses"][indexes[0]]
                        != self.proposer_tokenizer.pad_token_id
                    )
                    reward_tensor = torch.zeros(length, device=rewards.device)
                    token_reward_tensor = torch.zeros(length, device=rewards.device)
                    reward_tensor[real_length - 1] = final_reward
                    token_reward_tensor[: real_length - 1] = final_reward
                    new_batch[key].append(reward_tensor)
                    if "token_level_scores" not in new_batch:
                        new_batch["token_level_scores"] = []
                    new_batch["token_level_scores"].append(token_reward_tensor)
        for key, value in new_batch.items():
            new_batch[key] = torch.stack(value, dim=0)
        batch = DataProto.from_single_dict(new_batch)
        return batch

    def _proposer_val_reward(self, batch: DataProto, group_indexes: dict):
        new_batch = {}
        for _, indexes in group_indexes.items():
            for key, value in batch.batch.items():
                if key not in new_batch:
                    new_batch[key] = []
                if key not in ["rm_scores", "advantages"]:
                    new_batch[key].append(value[indexes[0]])
                else:
                    rewards = value[indexes]
                    rewards_index = torch.nonzero(rewards, as_tuple=True)
                    rewards_value = rewards[rewards_index]
                    if rewards_value.shape[0] == 0:
                        final_reward = 0
                    else:
                        final_reward = torch.mean(rewards_value)

                    length = batch.batch["responses"].shape[1]
                    real_length = torch.sum(
                        batch.batch["responses"][indexes[0]]
                        != self.proposer_tokenizer.pad_token_id
                    )
                    if key == "rm_scores":
                        reward_tensor = torch.zeros(length, device=rewards.device)
                        token_reward_tensor = torch.zeros(length, device=rewards.device)
                        reward_tensor[real_length - 1] = final_reward
                        token_reward_tensor[real_length - 1] = final_reward
                        new_batch[key].append(reward_tensor)
                        if "token_level_scores" not in new_batch:
                            new_batch["token_level_scores"] = []
                        new_batch["token_level_scores"].append(token_reward_tensor)
                    elif key == "advantages":
                        advantages = torch.zeros(length, device=rewards.device)
                        advantages[: real_length - 1] = final_reward
                        new_batch[key].append(advantages)
        for key, value in new_batch.items():
            new_batch[key] = torch.stack(value, dim=0)
        batch = DataProto.from_single_dict(new_batch)

        val_metrics = self._validate()
        ave_score = np.mean(list(val_metrics.values()))
        if self.proposer_ema:
            ema_prev = None
            ema_curr = None
            for val_score in self.val_score_history:
                if ema_prev is None:
                    ema_prev = val_score[1]
                else:
                    ema_prev = (
                        self.proposer_ema_decay * ema_prev
                        + (1 - self.proposer_ema_decay) * val_score[1]
                    )
            ema_curr = (
                self.proposer_ema_decay * ema_prev
                + (1 - self.proposer_ema_decay) * ave_score
            )
            final_reward = (ema_curr - ema_prev) * 100
        else:
            final_reward = (ave_score - self.val_score_history[-1][1]) * 100
        val_metrics["proposer_actor/reward"] = final_reward
        self.val_score_history.append((self.global_steps, ave_score))
        self.actor_val_score_history.append((self.global_steps, ave_score))

        length = batch.batch["responses"].shape[1]
        token_level_scores = []
        rm_scores = []
        for i in range(len(batch.batch["responses"])):
            real_length = torch.sum(
                batch.batch["responses"][i] != self.proposer_tokenizer.pad_token_id
            )
            reward_tensor = torch.zeros(length, device=batch.batch["responses"].device)
            token_reward_tensor = torch.zeros(
                length, device=batch.batch["responses"].device
            )
            if self.config.actor_rollout_ref.actor.get(
                "proposer_multiply_train_reward", False
            ):
                if self.config.actor_rollout_ref.actor.get(
                    "proposer_multiply_train_reward_adv", False
                ):
                    scores = batch.batch["advantages"][i]
                else:
                    scores = batch.batch["rm_scores"][i]
                reward_index = torch.nonzero(scores, as_tuple=True)[0]
                if len(reward_index) == 0:
                    instruction_reward = -1
                else:
                    instruction_reward = final_reward * scores[reward_index[-1]]
            else:
                instruction_reward = final_reward
            reward_tensor[real_length - 1] = instruction_reward
            token_reward_tensor[real_length - 1] = instruction_reward
            rm_scores.append(reward_tensor)
            token_level_scores.append(token_reward_tensor)

        batch.batch["rm_scores"] = torch.stack(rm_scores, dim=0)
        batch.batch["token_level_scores"] = torch.stack(token_level_scores, dim=0)
        return batch, val_metrics

    def _update_proposer(self, batch: DataProto, timing_raw: dict, metrics: dict):
        group_indexes = {}
        for i, uid in enumerate(batch.non_tensor_batch["uid"]):
            if uid not in group_indexes:
                group_indexes[uid] = []
            group_indexes[uid].append(i)

        train_metric = self.config.actor_rollout_ref.actor.get(
            "proposer_train_metric", "train_reward"
        )

        if train_metric == "train_reward":
            batch = self._proposer_train_reward(batch, group_indexes)
        elif train_metric == "val_reward":
            batch, val_metrics = self._proposer_val_reward(batch, group_indexes)
            metrics.update(val_metrics)
        else:
            raise NotImplementedError

        if self.val_score_history[-1][1] == self.val_score_history[-2][1]:
            return None

        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()
        
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        with _timer("proposer_old_log_prob", timing_raw):
            old_log_prob = self.proposer_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)

        with _timer("proposer_ref", timing_raw):
            ref_log_prob = self.proposer_ref_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        with _timer("proposer_adv", timing_raw):
            # compute rewards. apply_kl_penalty if available
            if not self.config.actor_rollout_ref.actor.get("use_kl_loss", False):
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                for k, v in kl_metrics.items():
                    kl_metrics["proposer_" + k] = v
                    del kl_metrics[k]
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # compute advantages, executed on the driver process
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
            )
        with _timer("update_proposer", timing_raw):
            proposer_output = self.proposer_rollout_wg.update_actor(batch)
        keys = list(proposer_output.meta_info["metrics"].keys())
        for k in keys:
            proposer_output.meta_info["metrics"]["proposer_" + k] = (
                proposer_output.meta_info["metrics"][k]
            )
            del proposer_output.meta_info["metrics"][k]
        return proposer_output

    def _filter(self, batch: DataProto):
        """
        Filter out instruction-response pairs with too high or too low reward scores.
        """
        group_indexes = {}
        group_rewards = {}
        for i, (uid, rm_score) in enumerate(
            zip(batch.non_tensor_batch["uid"], batch.batch["rm_scores"])
        ):
            if uid not in group_indexes:
                group_indexes[uid] = []
                group_rewards[uid] = []
            group_indexes[uid].append(i)
            group_rewards[uid].append(
                rm_score[torch.nonzero(rm_score, as_tuple=True)[-1]]
            )
        threshold = self.config.trainer.get("filter_threshold", 0)
        lower_bound = self.config.trainer.get("filter_lower_bound", 0)
        upper_bound = self.config.trainer.get("filter_upper_bound", 1.0)

        keep_indexes = []
        for uid in group_rewards:
            group_rewards[uid] = [reward >= threshold for reward in group_rewards[uid]]
            group_rewards[uid] = torch.mean(torch.cat(group_rewards[uid]).float())
            if group_rewards[uid] >= lower_bound and group_rewards[uid] <= upper_bound:
                keep_indexes.extend(group_indexes[uid])

        batch = batch.slice(keep_indexes)
        return batch, keep_indexes

    def fit(self):
        """
        The training loop of CPMobius.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        val_diff = []
        neg_val_diff = []
        val_diff_dict = {}
        neg_val_diff_count = 0

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            ave_score = np.mean(list(val_metrics.values()))
            self.val_score_history.append((0, ave_score))
            self.actor_val_score_history.append((0, ave_score))
            val_diff.append(0)
            val_diff_dict["val/diff"] = val_diff[-1]
            # if self.config.trainer.get('save_load_actor', False):
            #     self._save_checkpoint_actor()
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        proposer_batches = []

        

        for epoch in range(self.config.trainer.total_epochs):
            # for batch_dict in self.train_dataloader:
            timing_raw = {}
            with _timer("step", timing_raw):
                final_batch = None
                final_proposer_batch = None
                metrics = {}
                for i in range(100000):
                    metrics.update(val_diff_dict)
                    batch, proposer_batch = self._generate_instructions(
                        self.config.data.train_batch_size
                    )

                    # pop those keys for generation
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"]
                    )

                    # generate a batch
                    with _timer("gen", timing_raw):
                        print("Generating responses")
                        
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    proposer_batch.non_tensor_batch["uid"] = batch.non_tensor_batch[
                        "uid"
                    ]
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    proposer_batch = proposer_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)
                    
                    # compute rm score)
                    if self.use_rm:
                        with _timer("rm", timing_raw):
                            print("Computing rm scores")
                            rm_score = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_score)
                            proposer_batch = proposer_batch.union(rm_score)
                    reward_tensor = self.reward_fn(batch)
                    batch.batch["token_level_scores"] = reward_tensor
                    batch.batch["rm_scores"] = reward_tensor

                    batch, filtered_indexes = self._filter(batch)
                    proposer_batch = proposer_batch.slice(filtered_indexes)

                    if final_batch is None:
                        final_batch = batch
                        final_proposer_batch = proposer_batch
                    else:
                        final_batch = DataProto.concat([final_batch, batch])
                        final_proposer_batch = DataProto.concat(
                            [final_proposer_batch, proposer_batch]
                        )
                    if (
                        final_batch.batch.batch_size[0]
                        >= self.config.data.train_batch_size
                        * self.config.actor_rollout_ref.rollout.n
                    ):
                        final_batch = final_batch.slice(
                            list(
                                range(
                                    self.config.data.train_batch_size
                                    * self.config.actor_rollout_ref.rollout.n
                                )
                            )
                        )
                        final_proposer_batch = final_proposer_batch.slice(
                            list(
                                range(
                                    self.config.data.train_batch_size
                                    * self.config.actor_rollout_ref.rollout.n
                                )
                            )
                        )
                        break
                    else:
                        print(
                            f"Batch size: {final_batch.batch.batch_size[0]} / {self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n}"
                        )
                if final_batch is None:
                    raise ValueError("No valid batch generated")

                batch = final_batch
                proposer_batch = final_proposer_batch

                example_io = []
                example_instruction_io = []

                batch_size = len(batch.batch["input_ids"])
                sample_indices = np.random.choice(
                    batch_size, size=min(3, batch_size), replace=False
                )

                # Decode and store the sampled examples
                for idx in sample_indices:
                    response_decoded_text = self.tokenizer.decode(
                        batch.batch["input_ids"][idx], skip_special_tokens=True
                    )
                    example_io.append(response_decoded_text)
                metrics["example_io"] = example_io

                instruction_decoded_text = self.tokenizer.decode(
                    proposer_batch.batch["input_ids"][0], skip_special_tokens=True
                )
                example_instruction_io.append(instruction_decoded_text)
                metrics["example_instruction_io"] = example_instruction_io
                print(example_instruction_io)
                print(example_io)
                
                import swanlab
                instructions_table = swanlab.echarts.Table()
                instructions_table.add(["Instructions"], [[ex] for ex in example_instruction_io])
                responses_table = swanlab.echarts.Table()
                responses_table.add(["Responses"], [[ex] for ex in example_io])
                swanlab.log(
                    {
                        "instructions_examples": instructions_table,
                        "responses_examples": responses_table,
                    },
                    step=self.global_steps,
                )

                print(
                    f"Final batch size: {final_batch.batch.batch_size[0]} / {self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n}"
                )
                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                reorder_index = self._balance_batch(batch, metrics=metrics)
                proposer_batch.reorder(reorder_index)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()
                
                # recompute old_log_probs
                with _timer("old_log_prob", timing_raw):
                    print("Computing old log prob")
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer("ref", timing_raw):
                        print("Computing ref log prob")
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer("values", timing_raw):
                        print("Computing values")
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer("adv", timing_raw):

                    # compute rewards. apply_kl_penalty if available
                    if not self.config.actor_rollout_ref.actor.get(
                        "use_kl_loss", False
                    ):
                        batch, kl_metrics = apply_kl_penalty(
                            batch,
                            kl_ctrl=self.kl_ctrl,
                            kl_penalty=self.config.algorithm.kl_penalty,
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch[
                            "token_level_scores"
                        ]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                    )
                    proposer_batch.batch["advantages"] = batch.batch["advantages"]

                # update critic
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(
                        critic_output.meta_info["metrics"]
                    )
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer("update_actor", timing_raw):
                        if val_diff[-1] > 0:
                            print(
                                "Validation average score is positive, saving actor checkpoint"
                            )
                            self._save_checkpoint_actor()
                            neg_val_diff_count = 0
                        print("Updating actor")
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info["metrics"]
                    )
                    metrics.update(actor_output_metrics)

                proposer_batches.append(proposer_batch)
                proposer_train_begin_step = self.config.actor_rollout_ref.actor.get(
                    "proposer_train_begin_step", 0
                )
                if self.global_steps == proposer_train_begin_step:
                    val_metrics = self._validate()
                    ave_score = np.mean(list(val_metrics.values()))
                    self.val_score_history.append((self.global_steps, ave_score))
                    self.actor_val_score_history.append((self.global_steps, ave_score))
                    proposer_batches = []
                if (
                    self.global_steps
                    % self.config.trainer.get("proposer_update_interval", 1)
                    == 0
                    and self.global_steps > proposer_train_begin_step
                ):
                    with _timer("update_proposer", timing_raw):
                        print("Updating proposer")
                        proposer_output = self._update_proposer(
                            DataProto.concat(proposer_batches), timing_raw, metrics
                        )
                        
                        current_ave_score = self.actor_val_score_history[-1][-1]
                        ave_score_diff = (
                            current_ave_score - self.actor_val_score_history[-2][-1]
                        )
                        val_diff.append(ave_score_diff)
                        val_diff_dict["val/diff"] = val_diff[-1]
                        if ave_score_diff <= 0:
                            if self.global_steps >= self.config.trainer.get(
                                "save_load_begin_steps", 50
                            ):
                                neg_val_diff_count += 1
                            neg_val_diff.append(ave_score_diff)
                            removed_entry = self.actor_val_score_history.pop()
                            print(f"Removed validation score entry: {removed_entry}")
                        
                        print(val_diff)
                        print(neg_val_diff)
                        print(
                            f"actor_val_score_history: {self.actor_val_score_history}"
                        )
                        print(f"val_score_history: {self.val_score_history}")
                        print(
                            f"Current negative validation difference count: {neg_val_diff_count}"
                        )
                        
                        if self.config.trainer.get("save_load_actor", False):
                            if (
                                self.global_steps
                                >= self.config.trainer.get("save_load_begin_steps", 50)
                                and neg_val_diff_count
                                == self.config.trainer.get("save_load_interval", 1)
                                and np.signbit(ave_score_diff)
                            ):
                                print(
                                    "Validation average score is negative, loading actor checkpoint"
                                )
                                self._load_checkpoint_actor()
                                neg_val_diff_count = 0
                                
                                print(
                                    f"Current validation score history: {self.actor_val_score_history}"
                                )
                    proposer_batches = []
                    if proposer_output is not None:
                        proposer_output_metrics = reduce_metrics(
                            proposer_output.meta_info["metrics"]
                        )
                        metrics.update(proposer_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        ave_score = np.mean(list(val_metrics.values()))
                        self.actor_val_score_history.append(
                            (self.global_steps, ave_score)
                        )
                        current_ave_score = self.actor_val_score_history[-1][-1]
                        ave_score_diff = (
                            current_ave_score - self.actor_val_score_history[-2][-1]
                        )
                        val_diff.append(ave_score_diff)
                        if ave_score_diff < 0:
                            print(
                                "Validation average score is negative, loading actor checkpoint"
                            )
                            self._load_checkpoint_actor()
                            removed_entry = self.actor_val_score_history.pop()
                            print(f"Removed validation score entry: {removed_entry}")
                    metrics.update(val_metrics)

                if (
                    self.config.trainer.save_freq > 0
                    and self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            metrics.update(
                compute_data_metrics(batch=batch, use_critic=self.use_critic)
            )
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            self.global_steps += 1

            if self.global_steps >= self.total_training_steps:

                # perform validation after training
                if self.val_reward_fn is not None:
                    val_metrics = self._validate()
                    pprint(f"Final validation metrics: {val_metrics}")
                    logger.log(data=val_metrics, step=self.global_steps)
                return

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        proposer_local_path = os.path.join(local_global_step_folder, "proposer")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps
        )
        self.proposer_rollout_wg.save_checkpoint(
            proposer_local_path, None, self.global_steps
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps
            )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if not (
                self.config.trainer.resume_from_path and global_step_folder is not None
            ):
                assert isinstance(
                    self.config.trainer.resume_mode, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_mode
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        proposer_path = os.path.join(global_step_folder, "proposer")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        print(f"Actor loaded from {actor_path}")
        self.proposer_rollout_wg.load_checkpoint(proposer_path)
        print(f"Proposer loaded from {proposer_path}")
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)

    def _save_checkpoint_actor(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        temporary_actor_local_path = os.path.join(
            self.config.trainer.default_local_dir, "best_actor_checkpoint"
        )

        # proposer_local_path = os.path.join(local_global_step_folder, 'proposer')
        actor_local_path = os.path.join(temporary_actor_local_path, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, "best_actor_checkpoint", "actor"
            )
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, None)
        # self.proposer_rollout_wg.save_checkpoint(proposer_local_path, None, self.global_steps)

        if self.use_critic:
            critic_local_path = os.path.join(temporary_actor_local_path, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, "temp_critic_checkpoint"
                )
            )
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, None)

        # latest checkpointed iteration tracker (for atomic usage)
        local_best_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "best_checkpointed_iteration.txt"
        )
        with open(local_best_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint_actor(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            temp_actor_folder = os.path.join(checkpoint_folder, "best_actor_checkpoint")
            # global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
        print(f"Load from checkpoint folder: {temp_actor_folder}")

        actor_path = os.path.join(temp_actor_folder, "actor")
        if not os.path.exists(actor_path):
            # raise FileNotFoundError(f"Actor checkpoint not found at: {actor_path}")
            return
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        print(f"Actor loaded from {actor_path}")

