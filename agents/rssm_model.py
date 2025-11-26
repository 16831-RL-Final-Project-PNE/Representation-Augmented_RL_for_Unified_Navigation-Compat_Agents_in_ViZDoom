import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torch.distributions as td

RSSMDiscState = namedtuple('RSSMDiscState', ['logit', 'stoch', 'deter'])
class RSSMDiscrete(nn.Module):
    def __init__(
        self,
        action_size,
        rssm_node_size,
        embedding_size,  
        class_size = 20,
        category_size = 20,
        deter_size = 200,
        stoch_size = 20,
    ):
        nn.Module.__init__(self)
        self.action_size = action_size
        self.node_size = rssm_node_size
        self.embedding_size = embedding_size
        self.class_size = class_size
        self.category_size = category_size
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.state_action_embedder = nn.Sequential(
            nn.Linear(self.stoch_size + self.action_size, self.deter_size), 
            nn.ELU()
            )

        # p(z_t|h_t)
        self.temporal_prior = nn.Sequential(
            nn.Linear(self.deter_size, self.node_size), 
            nn.ELU(), 
            nn.Linear(self.node_size, self.stoch_size)
            )

        # q(z_t|h_t,o_t)
        self.temporal_posterior = nn.Sequential(
            nn.Linear(self.deter_size + self.embedding_size, self.node_size),
            nn.ELU(),
            nn.Linear(self.node_size, self.stoch_size)
            )

    def get_model_state(self, rssm_state):
        return torch.cat((rssm_state.deter, rssm_state.stoch), dim=-1)

    def get_stoch_state(self, logit: torch.Tensor):
        shape = logit.shape
        logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
        dist = td.OneHotCategorical(logits=logit)
        stoch = dist.sample()
        stoch += dist.probs - dist.probs.detach()
        return torch.flatten(stoch, start_dim=-2, end_dim=-1)  # (B, C, H, W) -> (B, C*H*W)

    def get_dist(self, rssm_state):
        shape = rssm_state.logit.shape
        logit = torch.reshape(rssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms):
        state_action_embed = self.state_action_embedder(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        prior_logit = self.temporal_prior(deter_state)
        prior_stoch_state = self.get_stoch_state(prior_logit)
        prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)
        return prior_rssm_state

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor(self.get_model_state(rssm_state))
            rssm_state = self.rssm_imagine(action, rssm_state, nonterms=torch.ones(1, dtype=torch.bool, device=rssm_state.stoch.device))
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_stack_states(self, rssm_states, dim):
        return RSSMDiscState(
            torch.stack([state.logit for state in rssm_states], dim=dim),
            torch.stack([state.stoch for state in rssm_states], dim=dim),
            torch.stack([state.deter for state in rssm_states], dim=dim),
        )

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        posterior_logit = self.temporal_posterior(x)
        posterior_stoch_state = self.get_stoch_state(posterior_logit)
        posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []

        for t in range(seq_len):
            prev_action = action[t]*nonterms[t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[t], prev_action, nonterms[t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=0)
        post = self.rssm_stack_states(posteriors, dim=0)
        return prior, post

    def _init_rssm_state(self, batch_size, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = next(self.parameters()).device
        return RSSMDiscState(
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.stoch_size, **kwargs),
            torch.zeros(batch_size, self.deter_size, **kwargs),
        )

    def rssm_seq_to_batch(self, rssm_state, batch_size, seq_len):
        def seq_to_batch(sequence_data, batch_size, seq_len):
            """
            converts a sequence of length L and batch_size B to a single batch of size L*B
            """
            shp = tuple(sequence_data.shape)
            batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
            return batch_data

        return RSSMDiscState(
            seq_to_batch(rssm_state.logit[:seq_len], batch_size, seq_len),
            seq_to_batch(rssm_state.stoch[:seq_len], batch_size, seq_len),
            seq_to_batch(rssm_state.deter[:seq_len], batch_size, seq_len)
        )

    def rssm_batch_to_seq(self, rssm_state, batch_size, seq_len):
        def batch_to_seq(batch_data, batch_size, seq_len):
            """
            converts a single batch of size L*B to a sequence of length L and batch_size B
            """
            shp = tuple(batch_data.shape)
            seq_data = torch.reshape(batch_data, [seq_len, batch_size, *shp[1:]])
            return seq_data

        return RSSMDiscState(
            batch_to_seq(rssm_state.logit, batch_size, seq_len),
            batch_to_seq(rssm_state.stoch, batch_size, seq_len),
            batch_to_seq(rssm_state.deter, batch_size, seq_len)
        )

    def rssm_detach(self, rssm_state):
        return RSSMDiscState(rssm_state.logit.detach(), rssm_state.stoch.detach(), rssm_state.deter.detach())
