# train/rl_trainer.py
from collections import OrderedDict
from typing import Optional, Dict, List
from tqdm.auto import trange
import time

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from train.replay_buffer import RolloutBuffer
from eval.evaluation import evaluate_policy, EvalLogger
from agents.random_agent import RandomAgent
from agents.ppo_agent import PPOAgent
from configs.ppo_config import PPOConfig


class RLTrainer:
    """
    Generic RL trainer that can work with different agents.
    For now, it implements PPO training for PPOAgent.
    """

    def __init__(
        self,
        env,
        eval_env,
        agent_type: str,
        config: PPOConfig,
        device: Optional[str] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Observation shape from env: (frame_stack, 3, H, W)
        self.obs_shape = env.observation_shape
        self.n_actions = env.action_space_n

        self.agent_type = agent_type.lower()
        self.agent = self._build_agent(self.agent_type).to(self.device)

        # Optimizer: only needed for trainable agents (e.g., PPO).
        # RandomAgent has no parameters, so we skip creating an optimizer.
        if agent_type == "random":
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(
                self.agent.parameters(),
                lr=self.config.learning_rate,
            )

        self.buffer = RolloutBuffer(
            buffer_size=self.config.steps_per_iteration,
            obs_shape=self.obs_shape,
            device=self.device,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        self.eval_logger = EvalLogger()

        # Training progress
        self.total_envsteps = 0
        self.start_time = time.time()
        self.initial_return: Optional[float] = None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _build_agent(self, agent_type: str):
        """
        Simple agent factory. Extend this to add more agent types.
        """
        if agent_type == "ppo":
            return PPOAgent(
                obs_shape=self.obs_shape,
                n_actions=self.n_actions,
            )
        elif agent_type == "random":
            # Stateless random policy
            return RandomAgent(n_actions=self.n_actions)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert stacked frames from DoomEnv into a (1, C, H, W) float tensor in [0,1].
        Input:  obs is (T, 3, H, W) uint8 from DoomEnv._get_obs()
        Output: (1, T*3, H, W) float32 on self.device
        """
        obs_t = torch.from_numpy(obs)  # (T, 3, H, W)
        if obs_t.ndim != 4:
            raise ValueError(f"Expected obs with 4 dims (T,3,H,W), got {obs_t.shape}")
        t, c, h, w = obs_t.shape

        # Use reshape instead of view to handle non-contiguous tensors safely.
        obs_t = obs_t.reshape(1, t * c, h, w).float() / 255.0
        return obs_t.to(self.device)

    def collect_rollout(self) -> Dict[str, List[float]]:
        """
        Collect one rollout of length steps_per_iteration into the buffer.
        Also track per-episode returns and lengths for training metrics.
        """
        self.buffer.reset()
        obs = self.env.reset()

        train_returns: List[float] = []
        train_ep_lens: List[int] = []
        ep_ret = 0.0
        ep_len = 0

        # Use tqdm to show per-iteration progress over environment steps
        for step in trange(
            self.config.steps_per_iteration,
            desc="Collecting rollout",
            leave=False,
        ):
            obs_tensor = self._obs_to_tensor(obs)

            with torch.no_grad():
                actions, log_probs, values = self.agent.act(obs_tensor, deterministic=False)

            action = int(actions.item())
            value = float(values.item())
            log_prob = float(log_probs.item())

            next_obs, reward, done, _info = self.env.step(action)

            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            ep_ret += float(reward)
            ep_len += 1

            obs = next_obs
            if done:
                train_returns.append(ep_ret)
                train_ep_lens.append(ep_len)
                ep_ret = 0.0
                ep_len = 0
                obs = self.env.reset()

        # If final episode did not terminate within this rollout, record it as a partial episode.
        if ep_len > 0:
            train_returns.append(ep_ret)
            train_ep_lens.append(ep_len)

        # Bootstrap value for the last state
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            last_value = float(self.agent.get_value(obs_tensor).item())

        self.buffer.compute_returns_and_advantages(last_value)

        self.total_envsteps += self.config.steps_per_iteration

        return {
            "train_returns": train_returns,
            "train_ep_lens": train_ep_lens,
        }

    def update_policy(self) -> Dict[str, float]:
        """
        Run multiple PPO epochs over the collected rollout.
        Return a dictionary of average losses for logging.
        """

        # Random agent: no training, just return zeros for logging
        if self.agent_type == "random":
            return {
                "Loss_Policy": 0.0,
                "Loss_Value": 0.0,
                "Loss_Entropy": 0.0,
            }

        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Show PPO epoch progress with tqdm
        for _ in trange(
            self.config.ppo_epochs,
            desc="PPO update",
            leave=False,
        ):
            for batch in self.buffer.get_minibatches(self.config.batch_size):
                obs_batch = batch["observations"]  # (B, C, H, W)
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, values = self.agent.evaluate_actions(
                    obs_batch, actions
                )

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        logs = {
            "Loss_Policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "Loss_Value": float(np.mean(value_losses)) if value_losses else 0.0,
            "Loss_Entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }
        return logs

    def _log_iteration(
        self,
        iteration: int,
        train_returns: List[float],
        train_ep_lens: List[int],
        eval_returns: List[float],
        eval_ep_lens: List[int],
        train_logs: Dict[str, float],
    ) -> None:
        """
        Aggregate metrics and write to TensorBoard + stdout,
        using homework-style keys.
        """
        logs = OrderedDict()

        # Eval stats
        eval_returns_np = np.asarray(eval_returns, dtype=np.float32)
        eval_ep_lens_np = np.asarray(eval_ep_lens, dtype=np.float32)
        logs["Eval_AverageReturn"] = float(eval_returns_np.mean())
        logs["Eval_StdReturn"] = float(eval_returns_np.std())
        logs["Eval_MaxReturn"] = float(eval_returns_np.max())
        logs["Eval_MinReturn"] = float(eval_returns_np.min())
        logs["Eval_AverageEpLen"] = float(eval_ep_lens_np.mean())

        # Train stats
        train_returns_np = np.asarray(train_returns, dtype=np.float32)
        train_ep_lens_np = np.asarray(train_ep_lens, dtype=np.float32)
        logs["Train_AverageReturn"] = float(train_returns_np.mean())
        logs["Train_StdReturn"] = float(train_returns_np.std())
        logs["Train_MaxReturn"] = float(train_returns_np.max())
        logs["Train_MinReturn"] = float(train_returns_np.min())
        logs["Train_AverageEpLen"] = float(train_ep_lens_np.mean())

        logs["Train_EnvstepsSoFar"] = float(self.total_envsteps)
        logs["TimeSinceStart"] = float(time.time() - self.start_time)

        # Initial_DataCollection_AverageReturn (use first iteration's train return)
        if self.initial_return is None:
            self.initial_return = logs["Train_AverageReturn"]
        logs["Initial_DataCollection_AverageReturn"] = float(self.initial_return)

        # Add training losses
        logs.update(train_logs)

        # Print + TensorBoard
        for key, value in logs.items():
            print(f"{key} : {value}")
            self.writer.add_scalar(key, value, iteration)
        self.writer.flush()
        print("Done logging...\n")

        # Also push eval mean/std to EvalLogger for later plotting
        self.eval_logger.add(
            iteration,
            logs["Eval_AverageReturn"],
            logs["Eval_StdReturn"],
        )

    def save_model(self, tag: str) -> str:
        """
        Save agent checkpoint. `tag` is usually an iteration index or 'final'.
        Returns the full path to the saved file.
        """
        filename = f"{self.config.checkpoint_name}_{tag}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)

        torch.save(
            {
                "agent_state_dict": self.agent.state_dict(),
                "agent_type": self.agent_type,
                "obs_shape": self.obs_shape,
                "n_actions": self.n_actions,
                "config": self.config,
            },
            path,
        )
        print(f"Saved model checkpoint to {path}")
        return path

    def train(self) -> None:
        """
        Main training loop:
        - collect rollouts
        - update policy
        - evaluate periodically and log mean returns vs iteration
        """
        for iteration in range(self.config.total_iterations):
            print(f"\n\n********** Iteration {iteration} ************")

            # 1) collect rollout + train metrics
            rollout_info = self.collect_rollout()
            train_returns = rollout_info["train_returns"]
            train_ep_lens = rollout_info["train_ep_lens"]

            # 2) PPO update
            train_logs = self.update_policy()

            # 3) evaluation for logging
            eval_returns, eval_ep_lens = evaluate_policy(
                env=self.eval_env,
                agent=self.agent,
                num_episodes=self.config.eval_episodes,
                device=self.device,
                deterministic=True,
                return_raw=True,
            )

            # 4) aggregate + log
            self._log_iteration(
                iteration=iteration,
                train_returns=train_returns,
                train_ep_lens=train_ep_lens,
                eval_returns=eval_returns,
                eval_ep_lens=eval_ep_lens,
                train_logs=train_logs,
            )

        # Save final model
        self.save_model(tag="final")

        # Save eval log (for npz + plotting)
        self.eval_logger.save(self.config.eval_log_path)
        print(f"Saved eval log to {self.config.eval_log_path}")
        self.writer.close()
