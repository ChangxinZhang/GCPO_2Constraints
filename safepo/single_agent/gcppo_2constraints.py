# Copyright 2025 Changxin Zhang, NUDT. Released under the MIT License.
# ==============================================================================

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque

import numpy as np
try: 
    from isaacgym import gymutil
except ImportError:
    pass
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal

from safepo.common.buffer import MultiObjectiveVectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env, make_sa_isaac_env
from safepo.common.lagrange import Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import GC2ActorVCritic_2constrains
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params


default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.02,
    'batch_size': 64,
    'learning_iters': 40,
    'max_grad_norm': 40.0,
}

isaac_gym_specific_cfg = {
    'total_steps': 200000000, # 100000000
    'steps_per_epoch': 38400, #   32768 393216
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': 0.016,
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
}

def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')


    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.task, seed=None)
        config = default_cfg

    else:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        sim_params = parse_sim_params(args, cfg_env, None)
        sim_params.physx.num_threads = min(args.num_threads, 4) 
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024  
        
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = GC2ActorVCritic_2constrains(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)
    reward_actor_optimizer = torch.optim.Adam(policy.reward_actor.parameters(), lr=3e-4)
    cost1_actor_optimizer = torch.optim.Adam(policy.cost1_actor.parameters(), lr=3e-4)
    cost2_actor_optimizer = torch.optim.Adam(policy.cost2_actor.parameters(), lr=3e-4)
    reward_actor_scheduler = LinearLR(
        reward_actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    cost1_actor_scheduler = LinearLR(
        cost1_actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    cost2_actor_scheduler = LinearLR(
        cost2_actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=False,
    )
    
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=3e-4
    )
    cost1_critic_optimizer = torch.optim.Adam(
        policy.cost1_critic.parameters(), lr=3e-4
    )
    cost2_critic_optimizer = torch.optim.Adam(
        policy.cost2_critic.parameters(), lr=3e-4
    )

    # create the multi-objective vectorized on-policy buffer
    buffer = MultiObjectiveVectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
        num_objectives=3  # reward、cost1、cost2
    )
    # setup lagrangian multiplier
    lagrange1 = Lagrange(
        cost_limit=args.cost1_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )
    lagrange2 = Lagrange(
        cost_limit=args.cost2_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
    )

    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    cost1_deque = deque(maxlen=50)
    cost2_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost1_deque = deque(maxlen=50)
    eval_cost2_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.reward_actor)
    logger.setup_torch_saver(policy.cost1_actor)
    logger.setup_torch_saver(policy.cost2_actor)
    logger.log("Start with training.")
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost1, ep_cost2, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c1, value_c2 = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            torch.cuda.synchronize()
            try:
                next_obs, reward, cost1, cost2, terminated, truncated, info = env.step(action)
            except RuntimeError as e:
                print(f"Error during env.step(): {e}")
                print("Resetting environment...")
                obs, _ = env.reset()
                obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
                continue
            reward *= 1
            cost1 *= 1
            cost2 *= 1

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost1 += cost1.cpu().numpy() if args.task in isaac_gym_map.keys() else cost1
            ep_cost2 += cost2.cpu().numpy() if args.task in isaac_gym_map.keys() else cost2
            ep_len += 1
            next_obs, reward, cost1, cost2, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost1, cost2, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            rewards = torch.stack([reward, cost1, cost2], dim=1)
            values = torch.stack([value_r, value_c1, value_c2], dim=1)
            buffer.store(
                obs=obs,
                act=act,
                rewards=rewards,
                values=values,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c1 = torch.zeros(1, device=device)
                    last_value_c2 = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c1, last_value_c2 = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c1, last_value_c2 = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c1 = last_value_c1.unsqueeze(0)
                        last_value_c2 = last_value_c2.unsqueeze(0)
                    last_values = torch.stack([last_value_r, last_value_c1, last_value_c2], dim=1).squeeze(0)
                    if epoch_end or done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost1_deque.append(ep_cost1[idx])
                        cost2_deque.append(ep_cost2[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost1": np.mean(cost1_deque),
                                "Metrics/EpCost2": np.mean(cost2_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost1[idx] = 0.0
                        ep_cost2[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False
                    
                        # if done or time_out:
                        #     print('There exists done or time_out')

                    buffer.finish_path(
                        last_values=last_values, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost1, eval_cost2, eval_len = 0.0, 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c1, value_c2 = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost1, cost2, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost1 += cost1
                    eval_cost2 += cost2
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost1_deque.append(eval_cost1)
                eval_cost2_deque.append(eval_cost2)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost1": np.mean(eval_cost1),
                    "Metrics/EvalEpCost2": np.mean(eval_cost2),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs1 = logger.get_stats("Metrics/EpCost1")
        ep_costs2 = logger.get_stats("Metrics/EpCost2")
        lagrange1.update_lagrange_multiplier(ep_costs1)
        lagrange2.update_lagrange_multiplier(ep_costs2)

        c1_lr_record = cost1_actor_optimizer.param_groups[0]['lr']
        c2_lr_record = cost2_actor_optimizer.param_groups[0]['lr']
        r_lr_record = reward_actor_optimizer.param_groups[0]['lr']

        if ep_costs1 > args.cost1_limit:
            cost1_actor_optimizer.param_groups[0]['lr'] *= 1.2
        elif ep_costs2 > args.cost2_limit:
            cost2_actor_optimizer.param_groups[0]['lr'] *= 1.2
        else:
            reward_actor_optimizer.param_groups[0]['lr'] *= 1.2

        # update policy
        data = buffer.get()

        reward_dist = policy.reward_actor(data["obs"])
        cost1_dist = policy.cost1_actor(data["obs"])
        cost2_dist = policy.cost2_actor(data["obs"])
        r_mean = reward_dist.mean
        r_std = reward_dist.stddev
        c1_mean = cost1_dist.mean
        c1_std = cost1_dist.stddev
        c2_mean = cost2_dist.mean
        c2_std = cost2_dist.stddev

        r_var = r_std ** 2
        c1_var = c1_std ** 2
        c2_var = c2_std ** 2

        sum_inv_var = 1/r_var + 1/c1_var + 1/c2_var

        action_mean = (r_mean/r_var + c1_mean/c1_var + c2_mean/c2_var) / sum_inv_var
        action_std = torch.sqrt(1 / sum_inv_var)

        old_distribution = Normal(action_mean, action_std)
         

        # compute advantage
        advantage = data["advs"][:, 0] - lagrange1.lagrangian_multiplier * data["advs"][:, 1] - lagrange2.lagrangian_multiplier * data["advs"][:, 2]
        advantage /= (lagrange1.lagrangian_multiplier + lagrange2.lagrangian_multiplier + 1)

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_values"][:, 0],
                data["target_values"][:, 1],
                data["target_values"][:, 2],
                advantage,
                data["advs"][:, 0],
                data["advs"][:, 1],
                data["advs"][:, 2],
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for _ in range(config["learning_iters"]):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c1_b,
                target_value_c2_b,
                adv_b,
                adv_r_b,
                adv_c1_b,
                adv_c2_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost1_critic_optimizer.zero_grad()
                loss_c1 = nn.functional.mse_loss(policy.cost1_critic(obs_b), target_value_c1_b)
                cost2_critic_optimizer.zero_grad()
                loss_c2 = nn.functional.mse_loss(policy.cost2_critic(obs_b), target_value_c2_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost1_critic.parameters():
                        loss_c1 += param.pow(2).sum() * 0.001
                    for param in policy.cost2_critic.parameters():
                        loss_c2 += param.pow(2).sum() * 0.001

                reward_dist_b = policy.reward_actor(obs_b)
                cost1_dist_b = policy.cost1_actor(obs_b)
                cost2_dist_b = policy.cost2_actor(obs_b)
                r_mean = reward_dist_b.mean
                r_std = reward_dist_b.stddev
                c1_mean = cost1_dist_b.mean
                c1_std = cost1_dist_b.stddev
                c2_mean = cost2_dist_b.mean
                c2_std = cost2_dist_b.stddev

                r_var = r_std ** 2
                c1_var = c1_std ** 2
                c2_var = c2_std ** 2

                sum_inv_var = 1/r_var + 1/c1_var + 1/c2_var

                action_mean = (r_mean/r_var + c1_mean/c1_var + c2_mean/c2_var) / sum_inv_var
                action_std = torch.sqrt(1 / sum_inv_var)

                distribution = Normal(action_mean, action_std)

                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(ratio, 0.8, 1.2)
                # loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                loss_pi_r = -torch.min(ratio * adv_r_b, ratio_cliped * adv_r_b).mean() + 0.01 * torch.max(ratio * adv_c1_b, ratio_cliped * adv_c1_b).mean() + 0.01 * torch.max(ratio * adv_c2_b, ratio_cliped * adv_c2_b).mean()
                loss_pi_c1 = torch.max(ratio * adv_c1_b, ratio_cliped * adv_c1_b).mean() - 0.01 * torch.min(ratio * adv_r_b, ratio_cliped * adv_r_b).mean()
                loss_pi_c2 = torch.max(ratio * adv_c2_b, ratio_cliped * adv_c2_b).mean() - 0.01 * torch.min(ratio * adv_r_b, ratio_cliped * adv_r_b).mean()
                reward_actor_optimizer.zero_grad()
                cost1_actor_optimizer.zero_grad()
                cost2_actor_optimizer.zero_grad()
                total_loss = loss_pi_r + loss_pi_c1 + loss_pi_c2 + 2*loss_r + loss_c1 + loss_c2 \
                    if config.get("use_value_coefficient", False) \
                    else loss_pi_r + loss_pi_c1 + loss_pi_c2 + loss_r + loss_c1 + loss_c2
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost1_critic_optimizer.step()
                cost2_critic_optimizer.step()
                reward_actor_optimizer.step()
                cost1_actor_optimizer.step()
                cost2_actor_optimizer.step()

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost1_critic": loss_c1.mean().item(),
                        "Loss/Loss_cost2_critic": loss_c2.mean().item(),
                        "Loss/Loss_actor_r": loss_pi_r.mean().item(),
                        "Loss/Loss_actor_c1": loss_pi_c1.mean().item(),
                        "Loss/Loss_actor_c2": loss_pi_c2.mean().item(),
                    }
                )

            # new_distribution = policy.actor(data["obs"])
            reward_dist = policy.reward_actor(data["obs"])
            cost1_dist = policy.cost1_actor(data["obs"])
            cost2_dist = policy.cost2_actor(data["obs"])
            r_mean = reward_dist.mean
            r_std = reward_dist.stddev
            c1_mean = cost1_dist.mean
            c1_std = cost1_dist.stddev
            c2_mean = cost2_dist.mean
            c2_std = cost2_dist.stddev

            r_var = r_std ** 2
            c1_var = c1_std ** 2
            c2_var = c2_std ** 2

            sum_inv_var = 1/r_var + 1/c1_var + 1/c2_var

            action_mean = (r_mean/r_var + c1_mean/c1_var + c2_mean/c2_var) / sum_inv_var
            action_std = torch.sqrt(1 / sum_inv_var)

            new_distribution = Normal(action_mean, action_std)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > config["target_kl"]:
                break
        update_end_time = time.time()

        cost1_actor_optimizer.param_groups[0]['lr'] = c1_lr_record 
        cost2_actor_optimizer.param_groups[0]['lr'] = c2_lr_record 
        reward_actor_optimizer.param_groups[0]['lr'] =  r_lr_record 
        reward_actor_scheduler.step()
        cost1_actor_scheduler.step()
        cost2_actor_scheduler.step()


        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost1")
            logger.log_tabular("Metrics/EpCost2")
            logger.log_tabular("Metrics/EpLen")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost1")
                logger.log_tabular("Metrics/EvalEpCost2")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/StopIter", update_counts)
            logger.log_tabular("Train/KL", final_kl)
            logger.log_tabular("Train/LagragianMultiplier1", lagrange1.lagrangian_multiplier)
            logger.log_tabular("Train/LagragianMultiplier2", lagrange2.lagrangian_multiplier)
            logger.log_tabular("Train/LR_C1", cost1_actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Train/LR_C2", cost2_actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Train/LR_R", reward_actor_scheduler.get_last_lr()[0])
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost1_critic")
            logger.log_tabular("Loss/Loss_cost2_critic")
            logger.log_tabular("Loss/Loss_actor_r")
            logger.log_tabular("Loss/Loss_actor_c1")
            logger.log_tabular("Loss/Loss_actor_c2")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["advs"][:, 0].mean().item())
            logger.log_tabular("Value/Cost1Adv", data["advs"][:, 1].mean().item())
            logger.log_tabular("Value/Cost2Adv", data["advs"][:, 2].mean().item())

            logger.dump_tabular()
            if (epoch+1) % 100 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                if args.task not in isaac_gym_map.keys():
                    logger.save_state(
                        state_dict={
                            "Normalizer": env.obs_rms,
                        },
                        itr = epoch
                    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)