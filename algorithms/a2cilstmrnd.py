from torch import nn
from torch.autograd import Variable
import torch
import gym
import time
import numpy as np
from models.a2cilstmrnd import ActorCritic, RND
from tqdm import tqdm
from src.utils import *
from src.logger import Logger
import torch.nn.functional as F

class A2CiLSTMRNDAgent:
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

        # LSTM param
        self.hx = None
        self.cx = None

        # Models
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.load_models(load_model_path)

        # RND - Curious Modules
        self.rnd = RND(state_dim=2)

        # Optimizers
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = lr)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters())

        # Loggers
        self.actor_critic_logger = Logger("A2CiLSTMRND", "model")
        self.rnd_logger = Logger("A2CiLSTMRND", "rnd")

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
                self.hx = Variable(torch.zeros(1, 16))
                self.cx = Variable(torch.zeros(1, 16))

                # Reset
                episode_reward = 0
                episode_actor_critic_loss = 0
                episode_int_loss = 0

                advantage_total = []                
                
                while not is_done:
                    # Feed Policy network
                    probs, _, _, (hxn, cxn), obs_features = self.actor_critic((  t(state), (self.hx, self.cx) ))

                    # Choose sample accoding to policy
                    action_dist = torch.distributions.Categorical(probs = probs)
                    action = action_dist.sample()
                    action_ix = action.detach().data

                    # Update env
                    if self.actions:
                        next_state, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
                    else:
                        next_state, ext_reward, is_done, info = self.env.step(action_ix)

                    
                    # Advantage 
                    _, Qi, Qe, _, _ = self.actor_critic(( t(next_state), (hxn, cxn) ))
                    _, Vi, Ve, _, _ = self.actor_critic(( t(state), (self.hx, self.cx) ))

                    target_o, predictor_o, int_reward = self.rnd( t(state[0:2]) )
                    int_reward = torch.clamp(int_reward, min = -10, max = 0)
                    
                    advantage_ext = ext_reward + (1-is_done)*(self.gamma * Qe) - Ve 
                    advantage_int = int_reward + (1-is_done)*(self.gamma * Qi) - Vi 

                    advantage = advantage_ext + advantage_int 
                    #print('int rew: {}, Ae : {}, Ai {}'.format(int_reward, advantage_ext, advantage_int ))

                    # Update models
                    actor_critic_loss, int_loss = self.update_models(advantage, action_dist, action, probs, target_o, predictor_o )

                    # Record losses and reward
                    episode_actor_critic_loss += actor_critic_loss
                    episode_reward += ext_reward
                    episode_int_loss += int_loss/1e6

                    state = next_state
                    # LSTM update cell
                    self.hx = Variable(hxn.data)
                    self.cx = Variable(cxn.data)

                self.rnd.reset()
                self.actor_critic_logger.update(episode_actor_critic_loss, episode_reward, self.actor_critic, save_best = True, save_checkpoints = True)
                self.rnd_logger.update(episode_int_loss, episode_reward, self.rnd)
                pbar.update()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        finally:
            try:
                self.actor_critic_logger.exception_arisen(self.actor_critic)
            except:
                pass
        pbar.close()

    def update_models(self, advantage, action_dist, action, probs, target_o, predictor_o):
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

        # RND Update
        int_loss = F.mse_loss( target_o.detach(), predictor_o)
        self.rnd_optimizer.zero_grad()
        int_loss.backward()
        self.rnd_optimizer.step()
        
        return float(actor_critic_loss), float(int_loss)

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False
        self.hx = Variable(torch.zeros(1, 16))
        self.cx = Variable(torch.zeros(1, 16))

        while not is_done:
            # Feed Policy network
            probs, _, _, (hxn, cxn), obs_features = self.actor_critic((  t(state), (self.hx, self.cx) ))

            # Choose sample accoding to policy
            action_dist = torch.distributions.Categorical(probs = probs)
            action = action_dist.sample()
            action_ix = action.detach().data

            if self.actions:
                next_state, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
            else:
                next_state, ext_reward, is_done, info = self.env.step(action_ix)

            target_o, predictor_o, int_reward = self.rnd( t(state[0:2]) )

            int_reward = torch.clamp(int_reward, min = -1, max = 0)

            print('int rew: {}, ext rew : {} '.format(int_reward, ext_reward))

            state = next_state
            self.hx = Variable(self.hx.data)
            self.cx = Variable(self.cx.data)
            time.sleep(0.01)
            self.env.render()

        self.env.close()

