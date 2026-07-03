"""
Main entry point for SDAR (Confidence-Gated Teacher Distillation) training.
Reuses SkillSDRayTrainer for teacher forward pass, but replaces the SDL loss
with a confidence-gated distillation loss in the actor.
"""

import copy

import hydra
import ray
from omegaconf import OmegaConf


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_sdar(config)


def run_sdar(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = SDARTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class SDARTaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        sdar_cfg = config.algorithm.get("sdar", {})
        with open_dict(config):
            config.actor_rollout_ref.actor.use_sdar_loss = True
            config.actor_rollout_ref.actor.sdar_loss_coef = sdar_cfg.get("sdar_coef", 0.1)
            config.actor_rollout_ref.actor.sdar_gate_beta = sdar_cfg.get("gate_beta", 5.0)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.rlsd_utils import SkillProvider

        skills_dir = sdar_cfg.get("skills_dir", "skills/alfworld")
        skill_all = sdar_cfg.get("skill_all", False)
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        print(f"[SDAR] Loaded skills from {skills_dir}")
        print(f"[SDAR] Available skills: {list(skill_provider.skill_contents.keys())}")
        print(f"[SDAR] Task-to-skill mapping: {skill_provider.task_to_skill}")
        print(f"[SDAR] sdar_coef: {config.actor_rollout_ref.actor.sdar_loss_coef}")
        print(f"[SDAR] gate_beta: {config.actor_rollout_ref.actor.sdar_gate_beta}")

        traj_collector = SDARTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            skill_provider=skill_provider,
        )

        from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer

        trainer = SkillSDRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        original_validate = trainer._validate
        eval_with_skill = _as_bool(sdar_cfg.get("eval_with_skill", True))

        def dual_validate():
            traj_collector.eval_inject_skill = False
            metrics_no_skill = original_validate()
            prefixed_no = {
                f"val_no_skill/{k.replace('val/', '')}": v
                for k, v in metrics_no_skill.items()
            }

            if not eval_with_skill:
                return prefixed_no

            traj_collector.eval_inject_skill = True
            metrics_with_skill = original_validate()
            prefixed_with = {
                f"val_with_skill/{k.replace('val/', '')}": v
                for k, v in metrics_with_skill.items()
            }

            combined = {}
            combined.update(prefixed_with)
            combined.update(prefixed_no)
            combined.update(metrics_with_skill)
            return combined

        trainer._validate = dual_validate
        trainer.init_workers()
        trainer.fit()


class SDARTrajectoryCollector:
    """
    Plain SDAR rollout collector with optional skill injection during validation.

    Training remains no-skill rollout. Validation can be run twice by toggling
    ``eval_inject_skill`` so metrics include both with-skill and no-skill
    environments.
    """

    def __init__(self, config, tokenizer, processor=None, skill_provider=None):
        from agent_system.multi_turn_rollout import TrajectoryCollector

        self._collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
        self.skill_provider = skill_provider
        self.eval_inject_skill = False

    def _inject_skills_into_obs(self, obs):
        if self.skill_provider is None:
            return obs

        skill_obs = copy.copy(obs)
        obs_texts = obs.get("text", None)
        if obs_texts is None:
            return skill_obs

        skill_texts = list(obs_texts)
        for i, text in enumerate(skill_texts):
            if text is None:
                continue
            skill_text = self.skill_provider.get_privileged_info_from_prompt(text)
            if skill_text:
                skill_texts[i] = f"[Privileged Skill Information]\n{skill_text}\n\n{text}"
        skill_obs["text"] = skill_texts
        return skill_obs

    def __getattr__(self, name):
        return getattr(self._collector, name)

    def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
        if is_train or not self.eval_inject_skill:
            return self._collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )

        original_preprocess_batch = self._collector.preprocess_batch

        def skill_preprocess_batch(gen_batch, obs):
            return original_preprocess_batch(gen_batch, self._inject_skills_into_obs(obs))

        self._collector.preprocess_batch = skill_preprocess_batch
        try:
            return self._collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                is_train=is_train,
            )
        finally:
            self._collector.preprocess_batch = original_preprocess_batch


if __name__ == "__main__":
    main()
