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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import sys


from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, boxed, prime
# from verl.utils.reward_score.evaluation_utils.math_util import _last_boxed_only_string
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, RayGANTrainer

sys.setrecursionlimit(10000)

def _default_compute_score(data_source, solution_str, ground_truth):
    if data_source == 'openai/gsm8k':
        # return gsm8k.compute_score(solution_str, ground_truth)
        return boxed.compute_score(solution_str, ground_truth)
    elif 'deepscaler' in data_source[0]:
        sys.path.append('Your path to deepscaler')
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn(solution_str, ground_truth)
    elif 'numina' in data_source:
        return boxed.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    elif isinstance(data_source, list) and data_source[0] == 'prime':
        return prime.compute_score(solution_str, ground_truth, ['math' for _ in range(len(ground_truth))])
    else:
        raise NotImplementedError(f"data_source {data_source} not supported")

def _BofN_compute_score(response_dict: dict):

    reward_dict = {}
    max_response_counts = max(response_dict.values())
    max_responses = [k for k, v in response_dict.items() if v == max_response_counts]
    
    if len(max_responses) == 1 and max_responses[0] is not None:
        for response in response_dict:
            reward_dict[response] = 1 if response in max_responses else -1
    else:
        for response in response_dict:
            reward_dict[response] = -1
    
    if None in response_dict:
        reward_dict[None] = -1

    print(reward_dict)
    return reward_dict



class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, batched=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.batched = batched

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_sources = []
        solution_strs = []
        ground_truths = []
        valid_response_lengths = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            if self.batched:
                data_sources.append(data_source)
                solution_strs.append(sequences_str)
                ground_truths.append(ground_truth)
                valid_response_lengths.append(valid_response_length)
                continue

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if self.batched:
            scores = self.compute_score(data_sources, solution_strs, ground_truths)
            for i, score in enumerate(scores):
                reward_tensor[i, valid_response_lengths[i] - 1] = score
                if data_sources[i] not in already_print_data_sources:
                    already_print_data_sources[data_sources[i]] = 0
                if already_print_data_sources[data_sources[i]] < self.num_examine:
                    already_print_data_sources[data_sources[i]] += 1
                    print(solution_strs[i])

        return reward_tensor
    
class BofNRewardManager():
    """The Best of N reward manager.
    """

    def __init__(self, tokenizer, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.compute_score = compute_score or _BofN_compute_score
    
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        from collections import defaultdict

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        instruction_response_dict = defaultdict(lambda: defaultdict(int))

        sample_metas = []

        for data_index in range(len(data)):
            data_item = data[data_index]
            uid = data_item.non_tensor_batch['uid']

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            response_sequences = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            valid_response_sequences = math.last_boxed_only_string(response_sequences)

            sample_metas.append((uid, valid_response_sequences, valid_response_length))

            instruction_response_dict[uid][valid_response_sequences] += 1
        
        uid_to_score_dict = {}
        for uid, response_counter in instruction_response_dict.items():
            score_dict = self.compute_score(response_counter)
            uid_to_score_dict[uid] = score_dict

        if len(sample_metas) != len(data):
            raise ValueError(f"Batch size mismatch: {len(sample_metas)} vs {len(data)}")

        for reward_index, (uid, response_sequence, response_length) in enumerate(sample_metas):
            if response_length == 0:
                continue
            reward_position = response_length - 1
            score = uid_to_score_dict[uid].get(response_sequence, -1)
            reward_tensor[reward_index, reward_position] = score
        return reward_tensor
        



import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        Role.ProposerRollout: ray.remote(ActorRolloutRefWorker),
        Role.ProposerRefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
        Role.ProposerRollout: global_pool_id,
        Role.ProposerRefPolicy: global_pool_id
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    if config.reward_model.enable:
        reward_fn = RewardManager(tokenizer=tokenizer,
                                num_examine=0,
                                compute_score=compute_score,
                                batched=config.trainer.get("reward_manager_batched", False))
    elif config.reward_model.get("use_best_of_n", False):
        print("Use Best of N reward function")
        reward_fn = BofNRewardManager(tokenizer=tokenizer,
                                    compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer,
                                  num_examine=1,
                                  compute_score=compute_score,
                                  batched=config.trainer.get("reward_manager_batched", False))

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    if config.trainer.task == 'basic':
        trainer = RayPPOTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
    elif config.trainer.task == 'gan':
        proposer_tokenizer = hf_tokenizer(config.actor_rollout_ref.model.proposer_path)
        trainer = RayGANTrainer(config=config,
                                tokenizer=tokenizer,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn,
                                proposer_tokenizer=proposer_tokenizer)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
