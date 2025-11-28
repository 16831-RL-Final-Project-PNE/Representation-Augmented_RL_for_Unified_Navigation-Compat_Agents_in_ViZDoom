# agents/ppo_agent.py
from typing import Tuple, Dict, abstractmethod, Iterable
import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from .nn_dreamerv2_models import *
from configs.dreamerv2_config import DreamerV2Config
from .rssm_model import *
from .nn_actor_critic_dinov3 import DinoV3Encoder


class DiscreteActionModel(nn.Module):
    def __init__(
        self,
        action_size,
        deter_size,
        stoch_size,
        embedding_size,
        config):
        super().__init__()
        self.action_size = action_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.embedding_size = embedding_size

        self.layers = config.actor_layers
        self.node_size = config.actor_node_size
        self.train_noise = config.actor_train_noise
        self.eval_noise = config.actor_eval_noise
        self.expl_min = config.actor_expl_min
        self.expl_decay = config.actor_expl_decay
        self.expl_type = config.actor_expl_type
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self.deter_size + self.stoch_size, self.node_size)]
        model += [nn.ELU()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.node_size, self.node_size)]
            model += [nn.ELU()]
        model += [nn.Linear(self.node_size, self.action_size)]
        return nn.Sequential(*model) 

    def forward(self, model_state):
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach()
        return action, action_dist

    def get_action_dist(self, model_state):
        logits = self.model(model_state)
        return td.OneHotCategorical(logits=logits)
            
    def add_exploration(self, action: torch.Tensor, itr: int, mode='train'):
        if mode == 'train':
            expl_amount = self.train_noise
            expl_amount = expl_amount - itr/self.expl_decay
            expl_amount = max(self.expl_min, expl_amount)
        elif mode == 'eval':
            expl_amount = self.eval_noise
        else:
            raise NotImplementedError
            
        if self.expl_type == 'epsilon_greedy':
            if np.random.uniform(0, 1) < expl_amount:
                index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
                action = torch.zeros_like(action)
                action[:, index] = 1
            return action
        # TODO: In case we want to include RND

        raise NotImplementedError

class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in self.get_parameters(self.modules)]

    def __enter__(self):
        for param in self.get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

    @abstractmethod
    def get_parameters(self, model_list: list[nn.Module]):
        return [param for model in model_list for param in model.parameters()]

class DreamerV2Agent(nn.Module):
    """
    DreamerV2 agent wrapping an ActorCritic network.

    The RLTrainer interacts with this class ONLY through:
    - act(...)
    - get_value(...)
    - update(...)         # DreamerV2-specific training logic
    """

    def __init__(self, obs_shape, n_actions: int, config: DreamerV2Config, feat_dim: int = 256):
        """
        obs_shape: (frame_stack, 3, H, W) from DoomEnv.observation_shape
        n_actions: size of discrete action space
        config: DreamerV2Config
        """
        super().__init__()
        frame_stack, c, h, w = obs_shape
        assert c == 3, f"Expected 3 channels per frame (RGB), got {c}"
        in_channels = frame_stack * c

        self.action_size = n_actions
        deter_size = config.rssm_deter_size
        category_size = config.rssm_category_size
        class_size = config.rssm_class_size
        stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        model_state_size = stoch_size + deter_size

        self.config = config

        print("Building RSSM...")
        self.rssm = RSSMDiscrete(self.action_size, rssm_node_size, embedding_size, 
                    class_size=class_size, category_size=category_size, deter_size=deter_size, stoch_size=stoch_size)
        print("Building Action Model...")
        self.action_model = DiscreteActionModel(self.action_size, deter_size, stoch_size, embedding_size, config)
        print("Building Reward and Value...")
        self.reward_decoder = DenseModel((1,), model_state_size, dist='normal')
        self.value_model = DenseModel((1,), model_state_size, dist='normal')
        print("Building Target Value Model...")
        self.target_value_model = DenseModel((1,), model_state_size, dist='normal')
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        print("Building Discount Model...")
        self.discount_model = DenseModel((1,), model_state_size, dist='binary')
        print("Building Observation Encoder and Decoder...")

        # TODO: Change here with DinoV3 encoder
        if config.use_dino_v3:
            self.obs_encoder = DinoV3Encoder(
                model_name=config.dino_v3_model_name,
                out_dim=embedding_size,
                freeze_backbone=config.dino_v3_freeze_backbone,
            )
        else:
            self.obs_encoder = ObsEncoder(obs_shape[1:], embedding_size)
        self.obs_decoder = ObsDecoder(obs_shape[1:], model_state_size)

        self.models_map = {
            'world_models': [self.obs_encoder, self.rssm, self.reward_decoder, self.obs_decoder, self.discount_model],
            'action_models': [self.action_model],
            'value_models': [self.value_model],
            'actor_critic_models': [self.action_model, self.value_model],
        }

    @abstractmethod
    def get_parameters(self, model_list: list[nn.Module]):
        return [param for model in model_list for param in model.parameters()]

    def reset(self):
        device = next(self.parameters()).device
        self.prev_rssm_state = self.rssm._init_rssm_state(1, device=device)
        self.prev_action = torch.zeros(1, self.action_size, device=device)
        self.prev_nonterm = torch.zeros(1, dtype=torch.bool, device=device)

    # --- core API used during rollout / evaluation ---
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True):
        """
        obs: (B, C, H, W)
        Returns: actions, log_probs, values (all tensors with batch dimension)
        """
        # e = embed(o_t)
        embed = self.obs_encoder(torch.tensor(obs, dtype=torch.float32))
        # z_t = q(z_t|h_t,e), h_t = f(z_{t-1}, a_{t-1})
        _, posterior_rssm_state = self.rssm.rssm_observe(embed, self.prev_action, self.prev_nonterm, self.prev_rssm_state)
        # m_t = [h_t, z_t]
        model_state = self.rssm.get_model_state(posterior_rssm_state)
        # a_t = pi(a_t|m_t)
        action_dist = self.action_model.get_action_dist(model_state)
        if deterministic:
            idx = torch.argmax(action_dist.probs, dim=-1).long()
        else:
            idx = td.Categorical(probs=action_dist.probs).sample().long()
        # log_pi(a_t|m_t)
        actions = F.one_hot(idx, num_classes=self.action_size).float()
        log_probs = action_dist.log_prob(actions)
        # update prev_nonterm, prev_rssm_state, prev_action
        self.prev_nonterm = torch.ones(1, dtype=torch.bool, device=self.prev_nonterm.device)
        self.prev_rssm_state = posterior_rssm_state
        self.prev_action = actions

        # v_t = V(m_t)
        value = self.value_model(model_state).mean
        return actions, log_probs, value

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W)
        Returns: values (B,)
        """
        # e = embed(o_t)
        embed = self.obs_encoder(torch.tensor(obs, dtype=torch.float32))
        # z_t = q(z_t|h_t,e), h_t = f(z_{t-1}, a_{t-1})
        _, posterior_rssm_state = self.rssm.rssm_observe(embed, self.prev_action, self.prev_nonterm, self.prev_rssm_state)
        # m_t = [h_t, z_t]
        model_state = self.rssm.get_model_state(posterior_rssm_state)
        # v_t = V(m_t)
        value = self.value_model(model_state).mean
        return value

    def _obs_loss(self, obs_dist, obs):
        # L_obs = E[log p(o_hat_t | o_t)]
        obs_loss = -torch.mean(obs_dist.log_prob(obs))
        return obs_loss

    def _discount_loss(self, discount_dist, nonterms):
        discount_target = nonterms.float()
        # L_discount = E[log p(disc_t|m_t)], m_t = [h_t, z_t]
        discount_loss = -torch.mean(discount_dist.log_prob(discount_target))
        return discount_loss

    def _reward_loss(self, reward_dist, rewards):
        # L_reward = E[log p(r_t|m_t)], m_t = [h_t, z_t]
        reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        return reward_loss

    def _kl_loss(self, prior, posterior):
        # L_kl = E[log p(z_hat_t | z_t) - log q(z_hat_t | z_t,e)]
        prior_dist = self.rssm.get_dist(prior)
        post_dist = self.rssm.get_dist(posterior)
        if self.config.kl_use_kl_balance:
            alpha = self.config.kl_balance_scale
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.rssm.get_dist(self.rssm.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.rssm.get_dist(self.rssm.rssm_detach(prior))))
            if self.config.kl_use_free_nats:
                free_nats = self.config.kl_free_nats
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.config.kl_use_free_nats:
                free_nats = self.config.kl_free_nats
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss

    def representation_loss(self, obs, actions, rewards, nonterms):
        embed = self.obs_encoder(obs)                                         #t to t+seq_len   
        prev_rssm_state = self.rssm._init_rssm_state(self.config.batch_size)
        prior, posterior = self.rssm.rollout_observation(self.config.rssm_seq_len, embed, actions, nonterms, prev_rssm_state)
        post_model_state = self.rssm.get_model_state(posterior)                #t to t+seq_len   
        obs_dist = self.obs_decoder(post_model_state[:-1])                     #t to t+seq_len-1  
        reward_dist = self.reward_decoder(post_model_state[:-1])               #t to t+seq_len-1  
        discount_dist = self.discount_model(post_model_state[:-1])                #t to t+seq_len-1   
        
        obs_loss = self._obs_loss(obs_dist, obs[:-1])
        reward_loss = self._reward_loss(reward_dist, rewards[1:])
        discount_loss = self._discount_loss(discount_dist, nonterms[1:])
        prior_dist, post_dist, kl_loss = self._kl_loss(prior, posterior)

        # world model loss
        world_model_loss = self.config.kl_loss_scale * kl_loss + reward_loss + obs_loss + self.config.discount_loss_scale*discount_loss
        return {
            "world_model_loss": world_model_loss,
            "kl_loss": kl_loss,
            "obs_loss": obs_loss,
            "reward_loss": reward_loss,
            "discount_loss": discount_loss,
            "prior_dist": prior_dist,
            "post_dist": post_dist,
            "posterior": posterior,
        }
        #model_loss, kl_loss, obs_loss, reward_loss, pcont_loss, prior_dist, post_dist, posterior

    def critic_loss(self, imag_model_states, discount, lambda_returns):
        with torch.no_grad():
            model_states = imag_model_states[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value_model(model_states) 
        critic_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return critic_loss

    def actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        def compute_return(
                        reward: torch.Tensor,
                        value: torch.Tensor,
                        discount: torch.Tensor,
                        bootstrap: torch.Tensor,
                        lambda_: float = 0.95
                    ):
            """
            Compute the discounted reward for a batch of data.
            reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
            Bootstrap is [batch, 1]
            """
            next_values = torch.cat([value[1:], bootstrap[None]], 0)
            target = reward + discount * next_values * (1 - lambda_)
            timesteps = list(range(reward.shape[0] - 1, -1, -1))
            outputs = []
            accumulated_reward = bootstrap
            for t in timesteps:
                inp = target[t]
                discount_factor = discount[t]
                accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
                outputs.append(accumulated_reward)
            returns = torch.flip(torch.stack(outputs), [0])
            return returns

        lambda_returns = compute_return(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1])

        if self.config.actor_grad == 'reinforce':
            advantage_imag = (lambda_returns - imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage_imag

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
        discount = torch.cumprod(discount_arr[:-1], 0)
        policy_entropy = policy_entropy[1:].unsqueeze(-1)
        actor_loss = -torch.sum(torch.mean(discount * (objective + self.config.actor_entropy_scale * policy_entropy), dim=1)) 
        return actor_loss, discount, lambda_returns

    def actor_critic_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.rssm.rssm_detach(self.rssm.rssm_seq_to_batch(posterior, self.config.batch_size, self.config.rssm_seq_len - 1))
        
        with FreezeParameters(self.models_map['world_models']):
            imag_rssm_states, imag_log_prob, policy_entropy = self.rssm.rollout_imagination(self.config.rssm_horizon, self.action_model, batched_posterior)
        
        imag_model_states = self.rssm.get_model_state(imag_rssm_states)
        with FreezeParameters(self.models_map['world_models'] + [self.target_value_model] + [self.discount_model]):
            imag_reward_dist = self.reward_decoder(imag_model_states)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.target_value_model(imag_model_states)
            imag_value = imag_value_dist.mean
            discount_dist = self.discount_model(imag_model_states)
            discount_arr = self.config.gamma * torch.round(discount_dist.base_dist.probs)              #mean = prob(disc==1)

        actor_loss, discount, lambda_returns = self.actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        critic_loss = self.critic_loss(imag_model_states, discount, lambda_returns)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()

        target_info = {
            'min_targ':min_targ,
            'max_targ':max_targ,
            'std_targ':std_targ,
            'mean_targ':mean_targ,
        }

        return actor_loss, critic_loss, target_info

    def update(self, buffer, optimizer: torch.optim.Optimizer, config) -> Dict[str, float]:
        """
        Run PPO updates using trajectories stored in `buffer`.

        The buffer is expected to provide:
        - .get_minibatches(batch_size) → yields dicts with keys:
          ["observations", "actions", "log_probs", "advantages", "returns", "dones"]

        Returns a dict of scalar logs:
        {
            "Loss_Policy": float,
            "Loss_Value": float,
            "Loss_Entropy": float,
        }
        """
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Outer loop: epochs, with tqdm progress bar
        for epoch_idx in trange(
            config.epochs,
            desc="DreamerV2 update",
            leave=False,
        ):
            for i in range(config.collect_intervals):
                batch = buffer.sample_sequences(config.rssm_seq_len, config.batch_size)
                obs_batch = batch["observations"]  # (B, L, C, H, W) already on correct device
                actions = batch["actions"]
                rewards = batch["rewards"]
                dones = batch["dones"]
                nonterms = torch.logical_not(dones)

                if actions.dim() == 2:
                    actions = F.one_hot(actions, num_classes=self.action_size).float()

                if nonterms.dim() == 2:
                    nonterms = nonterms.unsqueeze(-1)

                if rewards.dim() == 2:
                    rewards = rewards.unsqueeze(-1)

                loss_dict = self.representation_loss(obs_batch, actions, rewards, nonterms)


                optimizer['world_model'].zero_grad()
                loss_dict['world_model_loss'].backward()

                grad_norm_model = torch.nn.utils.clip_grad_norm_(self.get_parameters(self.models_map['world_models']), self.config.grad_clip_norm)
                optimizer['world_model'].step()

                actor_loss, value_loss, target_info = self.actor_critic_loss(loss_dict['posterior'])

                optimizer['action_model'].zero_grad()
                optimizer['value_model'].zero_grad()

                actor_loss.backward()
                value_loss.backward()

                grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.get_parameters(self.models_map['action_models']), self.config.grad_clip_norm)
                grad_norm_value = torch.nn.utils.clip_grad_norm_(self.get_parameters(self.models_map['value_models']), self.config.grad_clip_norm)

                optimizer['action_model'].step()
                optimizer['value_model'].step()

                with torch.no_grad():
                    prior_ent = torch.mean(loss_dict['prior_dist'].entropy())
                    post_ent = torch.mean(loss_dict['post_dist'].entropy())

                policy_losses.append(actor_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(prior_ent.item())
        
        # Soft update target networks
        self.update_target()

        logs = {
            "Loss_Policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "Loss_Value": float(np.mean(value_losses)) if value_losses else 0.0,
            "Loss_Entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }
        return logs


    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.value_model.parameters(), self.target_value_model.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
    
    def get_state_dict(self):
        return {
            "RSSM": self.RSSM.state_dict(),
            "ObsEncoder": self.ObsEncoder.state_dict(),
            "ObsDecoder": self.ObsDecoder.state_dict(),
            "RewardDecoder": self.RewardDecoder.state_dict(),
            "ActionModel": self.ActionModel.state_dict(),
            "ValueModel": self.ValueModel.state_dict(),
            "DiscountModel": self.DiscountModel.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.RSSM.load_state_dict(state_dict["RSSM"])
        self.ObsEncoder.load_state_dict(state_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(state_dict["ObsDecoder"])
        self.RewardDecoder.load_state_dict(state_dict["RewardDecoder"])
        self.ActionModel.load_state_dict(state_dict["ActionModel"])
        self.ValueModel.load_state_dict(state_dict["ValueModel"])
        self.DiscountModel.load_state_dict(state_dict["DiscountModel"])
    