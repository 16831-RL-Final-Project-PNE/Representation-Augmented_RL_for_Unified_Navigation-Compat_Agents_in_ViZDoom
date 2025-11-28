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
from agents.dreamerv2_agent import DreamerV2Agent
from configs.ppo_config import PPOConfig
from configs.dreamerv2_config import DreamerV2Config

import imageio
from pathlib import Path



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
        config: PPOConfig | DreamerV2Config,
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
        elif agent_type == "dreamerv2":
            if isinstance(self.config.learning_rate, dict):
                lr_world, lr_actor, lr_value = self.config.learning_rate['world_model'], self.config.learning_rate['actor_model'], self.config.learning_rate['value_model']
            else:
                raise ValueError(f"Invalid learning rate type: {type(self.config.learning_rate)}")

            self.optimizer = {
                'world_model': torch.optim.Adam(
                    self.agent.get_parameters(self.agent.models_map['world_models']),
                    lr=lr_world,
                ),
                'action_model': torch.optim.Adam(
                    self.agent.get_parameters(self.agent.models_map['action_models']),
                    lr=lr_actor,
                ),
                'value_model': torch.optim.Adam(
                    self.agent.get_parameters(self.agent.models_map['value_models']),
                    lr=lr_value,
                ),
            }
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
                feat_dim=self.config.feat_dim,
                backbone=self.config.backbone,        # "cnn" / "dinov2" / "dinov3"
                freeze_backbone=self.config.freeze_backbone,
            )

        elif agent_type == "random":
            # Stateless random policy
            return RandomAgent(n_actions=self.n_actions)
        elif agent_type == "dreamerv2":
            return DreamerV2Agent(
                obs_shape=self.obs_shape,
                n_actions=self.n_actions,
                config=self.config,
            )
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

        if hasattr(self.agent, "reset"):
            self.agent.reset()

        # Use tqdm to show per-iteration progress over environment steps
        for step in trange(
            self.config.steps_per_iteration,
            desc="Collecting rollout",
            leave=False,
        ):
            obs_tensor = self._obs_to_tensor(obs).to(self.device)

            with torch.no_grad():
                actions, log_probs, values = self.agent.act(obs_tensor, deterministic=True)

            if self.agent_type == "dreamerv2":
                actions = actions.argmax(dim=-1)

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
                if hasattr(self.agent, "reset"):
                    self.agent.reset()

        # If final episode did not terminate within this rollout, record it as a partial episode.
        if ep_len > 0:
            train_returns.append(ep_ret)
            train_ep_lens.append(ep_len)

        # Bootstrap value for the last state
        obs_tensor = self._obs_to_tensor(obs).to(self.device)
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
        Perform one policy/value update step.

        This method is intentionally algorithm-agnostic:
        - For non-trainable agents (e.g., RandomAgent), this is a no-op that
          returns zero losses for logging.
        - For trainable agents (e.g., PPOAgent, future DrQv2Agent, etc.),
          the actual update logic is delegated to the agent via its `update`
          method.

        The agent is responsible for:
        - reading batches from the provided rollout buffer,
        - computing its own losses,
        - calling `optimizer.step()`,
        - and returning a dict of scalar logs.
        """
        # Non-learning agents: no update
        if self.agent_type == "random":
            return {
                "Loss_Policy": 0.0,
                "Loss_Value": 0.0,
                "Loss_Entropy": 0.0,
            }

        # Trainable agents are expected to implement `update(...)`
        if not hasattr(self.agent, "update"):
            raise AttributeError(
                f"Agent of type '{self.agent_type}' does not implement an "
                f"`update(buffer, optimizer, config)` method."
            )

        # Delegate the actual RL algorithm to the agent
        logs = self.agent.update(
            buffer=self.buffer,
            optimizer=self.optimizer,
            config=self.config,
        )

        # Ensure we always return a dict with the standard keys
        default_logs = {
            "Loss_Policy": 0.0,
            "Loss_Value": 0.0,
            "Loss_Entropy": 0.0,
        }
        if logs is None:
            return default_logs

        default_logs.update(logs)
        return default_logs

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

        if self.agent_type == "dreamerv2":
            agent_state_dict = self.agent.get_state_dict()
        else:
            agent_state_dict = self.agent.state_dict()

        torch.save(
            {
                "agent_state_dict": agent_state_dict,
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
        - optionally save checkpoints every N iterations
        """
        for iteration in range(self.config.total_iterations):
            print(f"\n\n********** Iteration {iteration} ************")

            # 1) collect rollout + train metrics
            rollout_info = self.collect_rollout()
            train_returns = rollout_info["train_returns"]
            train_ep_lens = rollout_info["train_ep_lens"]

            # 2) Agent update
            train_logs = self.update_policy()

            # 3) evaluation for logging
            eval_returns, eval_ep_lens = evaluate_policy(
                env=self.eval_env,
                agent=self.agent,
                num_episodes=self.config.eval_episodes,
                device=self.device,
                deterministic=self.config.eval_deterministic,
                return_raw=True,
                agent_type=self.agent_type,
            )

            if iteration % 20 == 0:  # cada 20 iteraciones
                gif_path = os.path.join(self.config.log_dir, f"episode_{iteration}.gif")
                self.save_episode_gif(self.eval_env, self.agent, gif_path)

            # 4) aggregate + log
            self._log_iteration(
                iteration=iteration,
                train_returns=train_returns,
                train_ep_lens=train_ep_lens,
                eval_returns=eval_returns,
                eval_ep_lens=eval_ep_lens,
                train_logs=train_logs,
            )

            # 5) optional periodic checkpoint saving
            if (
                getattr(self.config, "save_every", 0) > 0
                and (iteration + 1) % self.config.save_every == 0
            ):
                # use iteration index (1-based) in filename
                self.save_model(tag=f"iter_{iteration + 1}")

        # Save final model
        self.save_model(tag="final")

        # Save eval log (for npz + plotting)
        self.eval_logger.save(self.config.eval_log_path)
        print(f"Saved eval log to {self.config.eval_log_path}")
        self.writer.close()

    def save_episode_gif(self, env, agent, path: str, max_frames: int = 2000, deterministic: bool = True):
        """
        Run one full episode in `env` using `agent` and save it as a GIF to `path`.
        The environment must return RGB frames as observations.
        """

        frames = []
        obs = env.reset()

        if hasattr(agent, "reset"):
            agent.reset()

        for _ in range(max_frames):
            # Store frame (assumes obs is RGB shape (H,W,3))
            frames.append(obs)

            obs_tensor = self._obs_to_tensor(obs).to(self.device)

            with torch.no_grad():
                actions, _, _ = agent.act(obs_tensor, deterministic=deterministic)
                if self.agent_type == "dreamerv2":
                    actions = actions.argmax(dim=-1)

            action = int(actions.item())
            obs, reward, done, _info = env.step(action)

            if done:
                frames.append(obs)
                break

        # Save GIF
        Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, frames, fps=30)
        print(f"Saved episode GIF to {path}")

