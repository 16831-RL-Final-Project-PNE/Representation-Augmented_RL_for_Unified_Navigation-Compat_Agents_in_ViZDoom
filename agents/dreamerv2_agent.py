
from typing import Tuple, Dict, abstractmethod, Iterable
import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import os

from .nn_dreamerv2_models import *
from configs.dreamerv2_config import DreamerV2Config
from .rssm_model import *
from .nn_actor_critic_dinov3 import DinoV3Encoder


import imageio
import numpy as np
import torch

def save_obs_dist_gif(obs_dist, filename="dreamer_recon.gif", fps=6, batch_idx=0):
    """
    Guarda un GIF sampleado desde obs_dist sin consumir memoria GPU ni RAM.
    Procesa frame por frame en CPU.
    """

    # Asegurar dimensión correcta
    if hasattr(obs_dist, "mean"):
        # No extraigas mean completa → extraerás frame-by-frame
        pass
    else:
        raise ValueError("obs_dist debe ser distribución Normal.")

    # Dimensiones
    T = obs_dist.mean.shape[0]   # solo mira shape, no extraigas los valores
    C = obs_dist.mean.shape[2]
    H = obs_dist.mean.shape[3]
    W = obs_dist.mean.shape[4]

    writer = imageio.get_writer(filename, fps=fps)

    for t in range(T):
        # ---- sampleo de UN SOLO FRAME ----
        frame_mean = obs_dist.mean[t, batch_idx]  # (C,H,W) en GPU
        frame = frame_mean.detach().cpu().clamp(0,1).numpy()  # copiar SOLO este frame

        # convertir a uint8
        frame = (frame * 255).astype(np.uint8)

        # CHW → HWC
        if frame.shape[0] in [1,3]:
            frame = np.transpose(frame, (1,2,0))

        # monocanal → RGB
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        writer.append_data(frame)

    writer.close()

def save_obs_dist_img(obs_dist, filename="dreamer_recon.png"):
    """
    Guarda un GIF sampleado desde obs_dist sin consumir memoria GPU ni RAM.
    Procesa frame por frame en CPU.
    """

    # Asegurar dimensión correcta
    if hasattr(obs_dist, "mean"):
        # No extraigas mean completa → extraerás frame-by-frame
        pass
    else:
        raise ValueError("obs_dist debe ser distribución Normal.")

    # Dimensiones
    C = obs_dist.mean.shape[-3]
    H = obs_dist.mean.shape[-2]
    W = obs_dist.mean.shape[-1]

    # ---- sampleo de UN SOLO FRAME ----
    frame_mean = obs_dist.mean.squeeze(0)  # (C,H,W) en GPU
    frame = frame_mean.detach().cpu().clamp(0,1).numpy()  # copiar SOLO este frame

    # convertir a uint8
    frame = (frame * 255).astype(np.uint8)

    # CHW → HWC
    if frame.shape[0] in [1,3]:
        frame = np.transpose(frame, (1,2,0))

    imageio.imwrite(filename, frame)


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

        self.rssm = RSSMDiscrete(self.action_size, rssm_node_size, embedding_size, 
                    class_size=class_size, category_size=category_size, deter_size=deter_size, stoch_size=stoch_size)
        self.action_model = DiscreteActionModel(self.action_size, deter_size, stoch_size, embedding_size, config)
        self.reward_decoder = DenseModel((1,), model_state_size, dist='normal')
        self.value_model = DenseModel((1,), model_state_size, dist='normal')
        self.target_value_model = DenseModel((1,), model_state_size, dist='normal')
        self.target_value_model.load_state_dict(self.value_model.state_dict())
        self.discount_model = DenseModel((1,), model_state_size, dist='binary')

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
        self.prev_nonterm = torch.zeros(1, dtype=torch.bool, device=device).unsqueeze(-1)

    # --- core API used during rollout / evaluation ---
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = True, itr: int = None, ):
        """
        obs: (B, C, H, W)
        Returns: actions, log_probs, values (all tensors with batch dimension)
        """
        # e = embed(o_t)
        embed = self.obs_encoder(torch.tensor(obs, dtype=torch.float32))
        # z_t = q(z_t|h_t,e), h_t = f(z_{t-1}, a_{t-1})
        prior_rssm_state, posterior_rssm_state = self.rssm.rssm_observe(embed, self.prev_action, self.prev_nonterm, self.prev_rssm_state)
        # m_t = [h_t, z_t]
        model_state = self.rssm.get_model_state(posterior_rssm_state)
        prior_model_state = self.rssm.get_model_state(prior_rssm_state)
        obs_dist = self.obs_decoder(model_state)                     #t+1 to t+seq_len
        prior_obs_dist = self.obs_decoder(prior_model_state)         #t+1 to t+seq_len
        with torch.no_grad():
            if np.random.random() < 0.01:
                save_obs_dist_img(obs_dist, "posterior_act.png")
                save_obs_dist_img(prior_obs_dist, "prior_act.png")
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
        self.prev_nonterm = torch.ones(1, dtype=torch.bool, device=self.prev_nonterm.device).unsqueeze(-1)
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
        # obs_dist.log_prob(obs) has shape (T, B) or (B,)
        # and is SUM over all pixel dims (C,H,W)

        log_prob = obs_dist.log_prob(obs)  # sum over pixels
        num_pix = np.prod(obs.shape[-3:])  # C*H*W

        # # Normalize per pixel
        log_prob_per_pixel = log_prob / num_pix

        # We want mean over batch/time
        obs_loss = -log_prob_per_pixel.mean()

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

        raw_kl = torch.mean(
            torch.distributions.kl.kl_divergence(post_dist, prior_dist)
        )
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
        return prior_dist, post_dist, kl_loss, raw_kl

    def representation_loss(self, obs, actions, rewards, nonterms):
        # next_obs, actions, rewards, nonterms
        # next_obs t+1 to t+seq_len+1
        # actions t to t+seq_len
        # rewards t to t+seq_len
        # nonterms t to t+seq_len
        embed = self.obs_encoder(obs)                                         #t+1 to t+seq_len+1
        prev_rssm_state = self.rssm._init_rssm_state(self.config.batch_size)
        prior, posterior = self.rssm.rollout_observation(self.config.rssm_seq_len, embed, actions, nonterms, prev_rssm_state)
        post_model_state = self.rssm.get_model_state(posterior)                #t+1 to t+seq_len+1   
        obs_dist = self.obs_decoder(post_model_state[:-1])                     #t+1 to t+seq_len

        with torch.no_grad():
            save_obs_dist_gif(obs_dist, os.path.join(self.config.log_dir, "reconstruction_posterior.gif"), fps=6)

        reward_dist = self.reward_decoder(post_model_state[:-1])               #t+1 to t+seq_len   
        discount_dist = self.discount_model(post_model_state[:-1])             #t+1 to t+seq_len   
        
        obs_loss = self._obs_loss(obs_dist, obs[:-1])                           # t+1 to t+seq_len
        reward_loss = self._reward_loss(reward_dist, rewards[1:])               # t+1 to t+seq_len
        discount_loss = self._discount_loss(discount_dist, nonterms[1:])        # t+1 to t+seq_len
        prior_dist, post_dist, kl_loss, raw_kl = self._kl_loss(prior, posterior) # t+1 to t+seq_len

        # world model loss
        world_model_loss = self.config.reward_loss_scale * reward_loss + self.config.obs_loss_scale * obs_loss + self.config.discount_loss_scale * discount_loss + self.config.kl_loss_scale * kl_loss
        
        return {
            "world_model_loss": world_model_loss,
            "kl_loss": kl_loss,
            "obs_loss": obs_loss,
            "reward_loss": reward_loss,
            "discount_loss": discount_loss,
            "prior_dist": prior_dist,
            "post_dist": post_dist,
            "posterior": posterior,
            "raw_kl_loss": raw_kl,
        }

    def critic_loss(self, imag_model_states, discount, lambda_returns):
        with torch.no_grad():
            model_states = imag_model_states[:-1].detach()
            value_discount = discount.detach()
            value_target = lambda_returns.detach()

        value_dist = self.value_model(model_states) 
        critic_loss = -torch.mean(value_discount*value_dist.log_prob(value_target).unsqueeze(-1))
        return critic_loss

    def actor_loss(self, imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy):
        # imag_reward t+1 to t+seq_len+1
        # imag_value t+1 to t+seq_len+1
        # discount_arr t+1 to t+seq_len+1
        # imag_log_prob t+1 to t+seq_len+1
        # policy_entropy t+1 to t+seq_len+1
        def compute_lambda_returns(reward, value, discount, bootstrap, lambda_=0.95):
            """
            reward[t]    : shape [T, B]
            value[t]     : shape [T, B]
            discount[t]  : shape [T, B]   (gamma_t)
            bootstrap    : value at last step v_{T}

            Returns G[t] (shape [T, B])
            """
            T = reward.shape[0]
            returns = torch.zeros_like(reward)
            last = bootstrap   # G_T = v_T

            for t in reversed(range(T)):
                # λ-return equation from Dreamer v2
                last = reward[t] + discount[t] * ((1 - lambda_) * value[t] + lambda_ * last)
                returns[t] = last

            return returns

        # lambda_returns t+1 to t+seq_len
        lambda_returns = compute_lambda_returns(imag_reward[:-1], imag_value[:-1], discount_arr[:-1], bootstrap=imag_value[-1])
        if self.config.actor_grad == 'reinforce':
            # advantage_imag t+1 to t+seq_len
            advantage_imag = (lambda_returns - imag_value[:-1]).detach()
            objective = imag_log_prob[1:].unsqueeze(-1) * advantage_imag

        elif self.config.actor_grad == 'dynamics':
            objective = lambda_returns
        else:
            raise NotImplementedError

        if self.config.actor_with_discount:
            discount_arr = torch.cat([torch.ones_like(discount_arr[:1, :, :]), discount_arr[1:, :, :]], dim=0)
            discount = torch.cumprod(discount_arr[:-1, :, :], dim=0)
        else:
            discount = torch.ones_like(discount_arr[:-1, :, :])
        policy_entropy = policy_entropy[1:, :].unsqueeze(-1)
        weighted_policy_entropy = self.config.actor_entropy_scale * policy_entropy
        loss_total = objective + weighted_policy_entropy
        discounted_loss_total = discount * loss_total
        if np.random.rand() < 0.01:
            print("Imagined rewards: ", imag_reward[:, 0, 0])
            print("Loss total: ", loss_total[:, 0, 0])
            print("Discounted loss total: ", discounted_loss_total[:, 0, 0])
        actor_loss = -discounted_loss_total.sum(dim=0).mean()
        return actor_loss, discount, lambda_returns

    def actor_critic_loss(self, posterior):
        with torch.no_grad():
            batched_posterior = self.rssm.rssm_detach(self.rssm.rssm_seq_to_batch(posterior, self.config.batch_size, self.config.rssm_seq_len - 1))
        
        with FreezeParameters(self.models_map['world_models']):
            imag_rssm_states, imag_log_prob, policy_entropy, action_dist_probs = self.rssm.rollout_imagination(self.config.rssm_horizon, self.action_model, batched_posterior)
        
        imag_model_states = self.rssm.get_model_state(imag_rssm_states)
        with FreezeParameters(self.models_map['world_models'] + [self.target_value_model] + [self.discount_model]):
            imag_reward_dist = self.reward_decoder(imag_model_states)
            imag_reward = imag_reward_dist.mean
            imag_value_dist = self.target_value_model(imag_model_states)
            imag_value = imag_value_dist.mean
            discount_dist = self.discount_model(imag_model_states)
            discount_arr = self.config.gamma * discount_dist.mean

        actor_loss, discount, lambda_returns = self.actor_loss(imag_reward, imag_value, discount_arr, imag_log_prob, policy_entropy)
        critic_loss = self.critic_loss(imag_model_states, discount, lambda_returns)

        mean_target = torch.mean(lambda_returns, dim=1)
        max_targ = torch.max(mean_target).item()
        min_targ = torch.min(mean_target).item() 
        std_targ = torch.std(mean_target).item()
        mean_targ = torch.mean(mean_target).item()
        policy_entropy = torch.mean(policy_entropy).item()

        target_info = {
            'min_target':min_targ,
            'max_target':max_targ,
            'std_target':std_targ,
            'mean_target':mean_targ,
            'policy_entropy':policy_entropy,
            'policy_dist_probs':action_dist_probs.detach().squeeze(),
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

        representation_losses = []
        kl_losses = []
        obs_losses = []
        reward_losses = []
        discount_losses = []
        raw_kl_losses = []
        policy_dist_probs = []  # for logging
        mean_target = []
        max_target = []
        min_target = []
        std_target = []

        # Outer loop: epochs, with tqdm progress bar
        for epoch_idx in trange(
            config.epochs,
            desc="DreamerV2 update",
            leave=False,
        ):
            for batch in buffer.sample_sequences(config.rssm_seq_len, config.batch_size):
                obs_batch = batch["observations"]  # (B, T, C, H, W) already on correct device
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

                # # normalize rewards
                rewards = rewards / self.config.reward_scale

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

                policy_losses.append(actor_loss.detach().item())
                value_losses.append(value_loss.detach().item())
                entropy_losses.append(target_info['policy_entropy'])
                representation_losses.append(loss_dict['world_model_loss'].detach().item())
                kl_losses.append(loss_dict['kl_loss'].detach().item())
                obs_losses.append(loss_dict['obs_loss'].detach().item())
                reward_losses.append(loss_dict['reward_loss'].detach().item())
                discount_losses.append(loss_dict['discount_loss'].detach().item())
                raw_kl_losses.append(loss_dict['raw_kl_loss'].detach().item())
                policy_dist_probs.append(target_info['policy_dist_probs'].detach())
                mean_target.append(target_info['mean_target'])
                max_target.append(target_info['max_target'])
                min_target.append(target_info['min_target'])
                std_target.append(target_info['std_target'])
        
                # Soft update target networks
                self.update_target()

        print("Current action distribution:")
        print(torch.stack(policy_dist_probs, dim=0).mean(dim=0))

        logs = {
            "Loss_Policy": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "Loss_Value": float(np.mean(value_losses)) if value_losses else 0.0,
            "Loss_Entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "Loss_Representation": float(np.mean(representation_losses)) if representation_losses else 0.0,
            "Loss_KL": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "Loss_Obs": float(np.mean(obs_losses)) if obs_losses else 0.0,
            "Loss_Reward": float(np.mean(reward_losses)) if reward_losses else 0.0,
            "Loss_Discount": float(np.mean(discount_losses)) if discount_losses else 0.0,
            "Loss_RawKL": float(np.mean(raw_kl_losses)) if raw_kl_losses else 0.0,
            "mean_target": float(np.mean(mean_target)) if mean_target else 0.0,
            "max_target": float(np.mean(max_target)) if max_target else 0.0,
            "min_target": float(np.mean(min_target)) if min_target else 0.0,
            "std_target": float(np.mean(std_target)) if std_target else 0.0,
        }
        return logs


    def update_target(self):
        mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
        for param, target_param in zip(self.value_model.parameters(), self.target_value_model.parameters()):
            target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
    
    def get_state_dict(self):
        return {
            "RSSM": self.rssm.state_dict(),
            "ObsEncoder": self.obs_encoder.state_dict(),
            "ObsDecoder": self.obs_decoder.state_dict(),
            "RewardDecoder": self.reward_decoder.state_dict(),
            "ActionModel": self.action_model.state_dict(),
            "ValueModel": self.value_model.state_dict(),
            "DiscountModel": self.discount_model.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.rssm.load_state_dict(state_dict["RSSM"])
        self.obs_encoder.load_state_dict(state_dict["ObsEncoder"])
        self.obs_decoder.load_state_dict(state_dict["ObsDecoder"])
        self.reward_decoder.load_state_dict(state_dict["RewardDecoder"])
        self.action_model.load_state_dict(state_dict["ActionModel"])
        self.value_model.load_state_dict(state_dict["ValueModel"])
        self.discount_model.load_state_dict(state_dict["DiscountModel"])
    