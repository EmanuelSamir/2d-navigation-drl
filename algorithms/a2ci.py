from torch import nn
import torch
import gym
import time
import numpy as np
from models.a2ci_lstm import ActorCritic
from tqdm import tqdm
from src.utils import *
from src.logger import Logger


class A2CiAgent:
    def __init__(self, env, state_dim, action_dim, actions, 
                n_episodes = 1_000, 
                gamma = 0.999,
                load_model_path = None,
                lr = 1e-3):

        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma

        # Discrete actions
        self.actions = actions

        # Models
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.load_models(load_model_path)

        # Optimizers
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = lr)

        # Loggers
        self.actor_critic_logger = Logger("A2Ci", "model")

    def load_models(self, model_path = None):
        # ActorCritic loading
        if model_path:
            checkpoint = torch.load(model_path)
            self.actor_critic.load_state_dict(checkpoint["model_state_dict"])


    def train(self):
        pbar = tqdm(total=self.n_episodes, position=0, leave=True)
        try:
            for episode in range(self.n_episodes):
                # Reset environment
                state = self.env.reset()
                is_done = False

                # Reset
                episode_reward = 0
                episode_actor_critic_loss = 0

                advantage_total = []                
                
                while not is_done:
                    # Feed Policy network
                    probs, _ = self.actor_critic(t(state).to(self.device))

                    # Choose sample accoding to policy
                    action_dist = torch.distributions.Categorical(probs = probs)
                    action = action_dist.sample()
                    action_ix = action.detach().data

                    # Update env
                    if self.actions:
                        next_state, reward, is_done, info = self.env.step(self.actions[action_ix])
                    else:
                        next_state, reward, is_done, info = self.env.step(action_ix)

                    
                    # Advantage 
                    _, Qv = self.actor_critic(t(next_state).to(self.device))
                    _, Vv = self.actor_critic(t(state).to(self.device))
                    
                    advantage = reward + (1-is_done)*(self.gamma * Qv - Vv )
                    
                    # Update models
                    actor_critic_loss = self.update_models(advantage, action_dist, action, probs)

                    # Record losses and reward
                    episode_actor_critic_loss += actor_critic_loss
                    episode_reward += reward

                    state = next_state

                
                self.actor_critic_logger.update(episode_actor_critic_loss, episode_reward, self.actor_critic, save_best = True, save_checkpoints = True)
                pbar.update()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        finally:
            try:
                self.actor_critic_logger.exception_arisen(self.actor_critic)
            except:
                pass
        pbar.close()

    def update_models(self, advantage, action_dist, action, probs):
        beta = 1e-2
        zeta = 1e-2

        # Actor Critic update
        value_loss = zeta * advantage.pow(2).mean() 
        policy_loss = - action_dist.log_prob(action) * advantage.detach()
        entropy_loss = - beta * (action_dist.log_prob(action) * probs).mean()

        actor_critic_loss = value_loss + policy_loss + entropy_loss

        self.actor_critic_optimizer.zero_grad()
        actor_critic_loss.backward()
        self.actor_critic_optimizer.step()
        
        return float(actor_critic_loss)

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False

        while not is_done:
            # Feed Policy network
            probs, _ = self.actor_critic(t(state).to(self.device))

            # Choose sample accoding to policy
            action_dist = torch.distributions.Categorical(probs = probs)
            action = action_dist.sample()
            action_ix = action.detach().data

            if self.actions:
                next_state, reward, is_done, info = self.env.step(self.actions[action_ix])
            else:
                next_state, reward, is_done, info = self.env.step(action_ix)

            state = next_state
            time.sleep(0.01)
            self.env.render()

        self.env.close()

