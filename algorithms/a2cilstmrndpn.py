from torch import nn
from torch.autograd import Variable
import torch
import gym
import time
import numpy as np
from models.a2cilstmrndpn import ActorCritic, RND
from tqdm import tqdm
from src.utils import *
from src.logger_adhoc import Logger
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class A2CiLSTMRNDPNAgent:
    def __init__(self, env, state_dim, action_dim, actions, 
                n_episodes = 1_000, 
                gamma = 0.999,
                load_model_path = None,
                lr = 1e-3,
                lr_rnd = 5e-3,
                trial = False):

        # General parameters
        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gammas
        self.trial = trial

        # Discrete actions
        self.actions = actions

        # LSTM param
        self.hx = None
        self.cx = None

        # Models
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.load_models(load_model_path)

        # RND - Curious Module
        self.rnd = RND(state_dim=288) # 256 + 32

        # Optimizers
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = lr)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr = lr_rnd )

        # Loggers: TODO
        if not self.trial:
            self.actor_critic_logger = Logger("A2CiLSTMRNDPN", "model")
            self.rnd_logger = Logger("A2CiLSTMRNDPN", "rnd")

            now = datetime.now()
            now = now.strftime("%d%m%Y_%H-%M-%S")
            sum_fn = now + '_lr={} eps={}'.format(lr, n_episodes)
            
            # self.tb = SummaryWriter(log_dir = '../log/A2CiLSTMRNDPN/' + sum_fn)

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

                # LSTM
                self.hx = Variable(torch.zeros(1, 512))
                self.cx = Variable(torch.zeros(1, 512))

                # Reset
                episode_ext_reward = 0
                episode_int_reward = 0
                episode_actor_critic_loss = 0
                episode_int_loss = 0

                policy_loss_lst = []
                value_loss_lst = []
                entropy_loss_lst = []
                target_lst = []
                predictor_lst = []
                int_reward_lst = []
                ext_reward_lst = []
                action_lst = []
                
                while not is_done:
                    # Feed Policy network
                    probs, _, _, (hxn, cxn), _ = self.actor_critic((  t(state), (self.hx, self.cx) ))

                    # Choose sample accoding to policy
                    action_dist = torch.distributions.Categorical(probs = probs)
                    action = action_dist.sample()
                    action_ix = action.detach().data

                    action_lst.append(action_ix)

                    # Update env
                    if self.actions:
                        next_state, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
                    else:
                        next_state, ext_reward, is_done, info = self.env.step(action_ix)

                    
                    # Advantage 
                    _, Qi, Qe, _, features = self.actor_critic(( t(next_state), (hxn, cxn) ))
                    _, Vi, Ve, _, _ = self.actor_critic(( t(state), (self.hx, self.cx) ))


                    #target_o, predictor_o, int_reward = self.rnd( features)
                    #int_reward = torch.clamp(int_reward, min = -5, max = 5)
                    predictor_o = 0
                    target_o = 0
                    

                    advantage_ext = ext_reward + (1-is_done)*(self.gamma * Qe) - Ve 
                    #advantage_int = int_reward + (1-is_done)*(self.gamma * Qi) - Vi 

                    advantage = advantage_ext #+ advantage_int 
                    #print('int rew: {}, Ae : {}, Ai {}'.format(int_reward, advantage_ext, advantage_int ))
                    #print(' {} '.format(target_o - predictor_o))

                    # Update models
                    actor_critic_loss, int_loss, v_loss, pi_loss, ent_loss = self.update_models(advantage, action_dist, action, probs, target_o, predictor_o )

                    # Record losses and reward
                    episode_actor_critic_loss += actor_critic_loss
                    episode_ext_reward += ext_reward
                    #episode_int_reward += int_reward
                    
                    #episode_int_loss += int_loss

                    policy_loss_lst.append(pi_loss)
                    value_loss_lst.append(v_loss)
                    entropy_loss_lst.append(ent_loss)
                    #target_lst.append(target_o.detach())
                    #predictor_lst.append(predictor_o.detach())
                    #int_reward_lst.append(int_reward.detach())
                    ext_reward_lst.append(ext_reward)


                    state = next_state
                    # LSTM update cell
                    self.hx = Variable(hxn.data)
                    self.cx = Variable(cxn.data)

                # if episodic:
                # self.rnd.reset()
                self.actor_critic_logger.update(episode_actor_critic_loss, episode_ext_reward, self.actor_critic, save_best = True, save_checkpoints = True)
                # self.rnd_logger.update(episode_int_loss, episode_int_reward, self.rnd)
                # LOG
                # for name, weight in self.actor_critic.named_parameters():
                    # self.tb.add_histogram(name, weight, episode)

                # for name, weight in self.rnd.named_parameters():
                    # self.tb.add_histogram(name, weight, episode)

                # self.tb.add_histogram('policy_loss', torch.stack(policy_loss_lst), episode)
                # self.tb.add_histogram('value_loss', torch.stack(value_loss_lst), episode)
                # self.tb.add_histogram('entropy_loss', torch.stack(entropy_loss_lst), episode)
                # self.tb.add_histogram('target', torch.stack(target_lst), episode)
                # self.tb.add_histogram('predictor', torch.stack(predictor_lst), episode)
                # self.tb.add_histogram('int_reward', torch.stack(int_reward_lst), episode)
                # self.tb.add_histogram('ext_reward', torch.tensor(ext_reward_lst), episode)
                # self.tb.add_histogram('actions', torch.tensor(action_lst), episode)
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
        beta = 1e1
        zeta = 1e-1

        # Actor Critic update
        value_loss = zeta * advantage.pow(2).mean() 
        policy_loss = - action_dist.log_prob(action) * advantage.detach()

        entropy_loss = - beta * (action_dist.log_prob(action) * probs).mean()
        #print('log_action = {}, probs {}'.format(action_dist.log_prob(action), probs ) )
        #print(' VL = {}, PL = {}, EL = {}'.format(value_loss, policy_loss, entropy_loss))

        actor_critic_loss = value_loss + policy_loss + entropy_loss

        self.actor_critic_optimizer.zero_grad()
        actor_critic_loss.backward()
        self.actor_critic_optimizer.step()

        int_loss = 0

        # RND Update
        #int_loss = F.mse_loss( target_o.detach(), predictor_o)
        #self.rnd_optimizer.zero_grad()
        #int_loss.backward()
        #self.rnd_optimizer.step()
        
        return float(actor_critic_loss), float(int_loss), value_loss.detach(), policy_loss.detach(), entropy_loss.detach()

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False
        self.hx = Variable(torch.zeros(1, 512))
        self.cx = Variable(torch.zeros(1, 512))

        while not is_done:
            # Feed Policy network
            probs, _, _, (hxn, cxn), features = self.actor_critic((  t(state), (self.hx, self.cx) ))

            # Choose sample accoding to policy
            action_dist = torch.distributions.Categorical(probs = probs)
            action = action_dist.sample()
            action_ix = action.detach().data

            if self.actions:
                next_state, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
            else:
                next_state, ext_reward, is_done, info = self.env.step(action_ix)

            #target_o, predictor_o, int_reward = self.rnd( features)
            
            #int_reward = torch.clamp(int_reward, min = -5, max = 5)

            #print('int rew: {}, ext rew : {} '.format(int_reward, ext_reward))

            state = next_state
            self.hx = Variable(hxn.data)
            self.cx = Variable(cxn.data)

            #int_loss = F.mse_loss( target_o.detach(), predictor_o)
            #self.rnd_optimizer.zero_grad()
            #int_loss.backward()
            #self.rnd_optimizer.step()

            time.sleep(0.01)
            self.env.render()
        #self.rnd.reset()
        self.env.close()
