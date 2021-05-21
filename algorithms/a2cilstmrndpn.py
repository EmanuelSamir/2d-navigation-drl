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
from src.logger_special import LoggerSpecial
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class A2CiLSTMRNDPNAgent:
    def __init__(self, env, state_dim, action_dim, actions, 
                n_episodes = 1_000, 
                gamma = 0.999,
                gamma_rnd = 0.999,
                lr = 1e-3,
                lr_rnd = 5e-3,
                load_model_fn = None,
                trial = False,
                intrinsic_set = True,
                save_best = True,
                save_checkpoints = True,
                checkpoint_every = 50,
                n_opt = 20,
                zeta = 1e-1,
                beta = 1e1
                ):

        # General parameters
        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.gamma_rnd = gamma_rnd
        self.trial = trial
        self.save_best = save_best
        self.save_checkpoints = save_checkpoints
        self.checkpoint_every = checkpoint_every 
        self.intrinsic_set = intrinsic_set
        self.lr = lr
        self.lr_rnd = lr_rnd
        self.zeta = zeta
        self.beta = beta
        self.n_opt = n_opt

        self.episode = 0
        

        # Discrete actions
        self.actions = actions

        # LSTM param
        self.hx = None
        self.cx = None

        # Models
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = self.lr)
        self.load_models(load_model_fn)

        # RND - Curious Module
        self.rnd = RND(state_dim=288) # 256 + 32
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr = self.lr_rnd )
        self.features_ns = None


        # Loggers:
        if not self.trial:
            self.logger = Logger("A2CiLSTMRNDPN", 
                                self.save_best,
                                self.save_checkpoints,
                                self.checkpoint_every)

        # self.logger_special = LoggerSpecial("A2CiLSTMRNDPN")


    def load_models(self, model_fn = None):
        # ActorCritic loading
        if model_fn:
            try:
                model_path = os.path.join("../checkpoints", "A2CiLSTMRNDPN", model_fn)
                checkpoint = torch.load(model_path)
                self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
                self.actor_critic_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                raise Exception("Model filename is incorrect")


    def train(self, comment = ''):
        pbar = tqdm(total=self.n_episodes, position=0, leave=True)
        if not self.trial:
            self.logger.set_description(    comment,
                                            self.lr,
                                            self.lr_rnd,
                                            self.zeta,
                                            self.beta,
                                            self.n_episodes,
                                            self.gamma,
                                            self.actor_critic,
                                            self.rnd,
                                            self.actions)

        # self.logger_special.set_description(comment)

        try:
            for episode in range(self.n_episodes):
                
                # Reset environment 
                state = self.env.reset()
                is_done = False
                self.episode = episode

                # LSTM
                self.hx = Variable(torch.zeros(1, 512))
                self.cx = Variable(torch.zeros(1, 512))

                # Env
                steps = 0

                # Reset
                while not is_done:
                    # Feed Policy network
                    probs, _, _, (hxn, cxn), _ = self.actor_critic((  tn(state), (self.hx, self.cx) ))

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
                    _, Qi, Qe, _, self.features_ns = self.actor_critic(( tn(next_state), (hxn, cxn) ))
                    _, Vi, Ve, _, _ = self.actor_critic(( tn(state), (self.hx, self.cx) ))

                    _, _, int_reward = self.rnd(self.features_ns)
                    int_reward = torch.clamp(int_reward, min = 0, max = 8)


                    advantage_ext = ext_reward + (1-is_done)*(self.gamma * Qe) - Ve 
                    advantage_int = int_reward + (1-is_done)*(self.gamma_rnd * Qi) - Vi 

                    if self.intrinsic_set:
                        advantage = advantage_ext + advantage_int 
                    else:
                        advantage = advantage_ext
                        int_reward = torch.tensor([0])

                    # Update models
                    v_loss, pi_loss, ent_loss = self.update_models(advantage, action_dist, action, probs)

                    # Record losses and reward
                    if not self.trial:
                        self.logger.update( action_ix,
                                            int_reward.detach(),
                                            ext_reward,
                                            v_loss,
                                            ent_loss,
                                            pi_loss)

                    # self.logger_special.update(steps, self.features_ns.tolist())

                    # Env update
                    state = next_state
                    steps += 1

                    # LSTM update cell
                    self.hx = Variable(hxn.data)
                    self.cx = Variable(cxn.data)

                self.rnd.reset()
                if not self.trial:
                    self.logger.consolidate(steps, self.episode, self.actor_critic, self.actor_critic_optimizer, self.rnd)
                # self.logger_special.consolidate(episode)
                pbar.update()
            
            if not self.trial:
                self.logger.close()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        finally:
            if not self.trial:
                self.logger.exception_arisen(self.episode, self.actor_critic, self.actor_critic_optimizer)
            
        pbar.close()

    def update_models(self, advantage, action_dist, action, probs):
        # Actor Critic update
        value_loss = self.zeta * advantage.pow(2).mean() 
        policy_loss = - action_dist.log_prob(action) * advantage.detach()

        entropy_loss = - self.beta * (action_dist.log_prob(action) * probs).mean()
        #print('log_action = {}, probs {}'.format(action_dist.log_prob(action), probs ) )
        #print(' VL = {}, PL = {}, EL = {}'.format(value_loss, policy_loss, entropy_loss))

        actor_critic_loss = value_loss + policy_loss + entropy_loss

        self.actor_critic_optimizer.zero_grad()
        actor_critic_loss.backward()
        self.actor_critic_optimizer.step()

        # RND Update
        if self.intrinsic_set:
            x = self.features_ns.unsqueeze(0)
            xs = torch.repeat_interleave(x, self.n_opt, dim=0)

            for x_i in xs:
                t, p, _ = self.rnd(x_i)
                int_loss = F.binary_cross_entropy(p, t.detach())
                self.rnd_optimizer.zero_grad()
                int_loss.backward()
                self.rnd_optimizer.step()
        
        return value_loss.detach(), policy_loss.detach(), entropy_loss.detach()

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False
        self.hx = Variable(torch.zeros(1, 512))
        self.cx = Variable(torch.zeros(1, 512))

        while not is_done:
            # Feed Policy network
            probs, _, _, (hxn, cxn), _ = self.actor_critic((  tn(state), (self.hx, self.cx) ))

            #print('LSTM hxn mean {} std {} y cxn mean {} std {} '.format(torch.mean(hxn), torch.std(hxn), torch.mean(cxn), torch.std(cxn) ))

            # Choose sample accoding to policy
            action_dist = torch.distributions.Categorical(probs = probs)
            action = action_dist.sample()
            action_ix = action.detach().data

            if self.actions:
                next_state, ext_reward, is_done, info = self.env.step(self.actions[action_ix])
            else:
                next_state, ext_reward, is_done, info = self.env.step(action_ix)

            _, Qi, Qe, _, self.features_ns = self.actor_critic(( tn(next_state), (hxn, cxn) ))
            _, Vi, Ve, _, _ = self.actor_critic(( tn(state), (self.hx, self.cx) ))

            _, _, int_reward = self.rnd(self.features_ns)
            
            int_reward = torch.clamp(int_reward, min = 0, max = 8)

            advantage_ext = ext_reward + (1-is_done)*(self.gamma * Qe) - Ve 
            advantage_int = int_reward + (1-is_done)*(self.gamma_rnd * Qi) - Vi 

            advantage = advantage_ext + advantage_int

            print('int rew: {}, ext rew : {} '.format(int_reward, ext_reward))

            value_loss = self.zeta * advantage.pow(2).mean() 
            policy_loss = - action_dist.log_prob(action) * advantage.detach()

            entropy_loss = - self.beta * (action_dist.log_prob(action) * probs).mean()
            #print('log_action = {}, probs {}'.format(action_dist.log_prob(action), probs ) )
            #print(' VL = {}, PL = {}, EL = {}'.format(value_loss, policy_loss, entropy_loss))
            

            state = next_state
            self.hx = Variable(hxn.data)
            self.cx = Variable(cxn.data)

            # RND Update
            x = self.features_ns.unsqueeze(0)
            xs = torch.repeat_interleave(x, self.n_opt, dim=0)

            for x_i in xs:
                t, p, _ = self.rnd(x_i)
                int_loss = F.binary_cross_entropy(p, t.detach())
                self.rnd_optimizer.zero_grad()
                int_loss.backward()
                self.rnd_optimizer.step()

            time.sleep(0.01)
            self.env.render()
        #self.rnd.reset()
        self.env.close()
