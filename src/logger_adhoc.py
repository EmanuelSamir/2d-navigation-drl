import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, yaml, io
import pickle
from torch import save as tsave
import torch
from .utils import create_dir
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, algorithm, 
                    save_best,
                    save_checkpoints,
                    checkpoint_every,
                    name = 'default'):

        # tmp option in case it is only test
        ## Description
        ### Gamma
        ### Learning Rate
        ### Comment
        ### Model Graph
        ### RND Graph. Necessary
        ### Episodes set
        ### Episodes achieved
        ### Distribution of what was saved. 
        ### Description if it is based on an old job
        ### Action set used
        ## Checkpoints:
        ### Model: Save model parameters
        ### Optimizer: Save optimizer parameters
        ### RND Optimizer Parameters
        ## Data (save cummulative during episode and save after)
        ### Rewards Extrinsic and Intrinsic: Accumulative
        ### Losses: Accumulative
        ### Steps: Accumulative
        ## Log (save cummulative during episode and save after and flush)
        ### Weights neural networks: Histograms
        ### Action: Histograms
        ### Rewards Extrinsic and Intrinsic: Histograms
        ### Losses: Histograms
        ## Results
        ### Plots Accumulative
        
        # Logger parameters
        self.save_best = save_best 
        self.save_checkpoints = save_checkpoints
        self.checkpoint_every = checkpoint_every
     
        # Data
        self._intrinsic_rewards_accummulative = []
        self._extrinsic_rewards_accummulative = []
        self._value_losses_accummulative = [] 
        self._entropy_losses_accummulative = []
        self._policy_losses_accummulative = []
        self._steps_accummulative = []


        # Log
        self._actions = []
        self._intrinsic_rewards = []
        self._extrinsic_rewards = []
        self._value_losses = []
        self._entropy_losses = []
        self._policy_losses = []

        # Results


        # Datapath

        fn_date = datetime.now().strftime("_%m%d_%H-%M-%S")

        self.save_description_path = os.path.join("../description", algorithm, fn_date)
        create_dir(self.save_description_path)

        self.save_model_path = os.path.join("../checkpoints", algorithm, fn_date)
        create_dir(self.save_model_path)

        self.save_data_path = os.path.join("../data", algorithm, fn_date)
        create_dir(self.save_data_path)

        self.save_result_path = os.path.join("../results", algorithm, fn_date)
        create_dir(self.save_result_path)

        self.save_log_path = os.path.join("../log", algorithm, fn_date)
        create_dir(self.save_log_path)

        self.tb = SummaryWriter(log_dir = os.path.join(self.save_log_path, 'log'))

        self.episode = 0
        self.steps = 0

        self.name = name
        
        self.best_reward = -10000


    def set_description(self, comment,
                             lr,
                             lr_rnd,
                             zeta,
                             beta,
                             eps,
                             gamma,
                             model,
                             rnd,
                             actions,
                             old_fn):

        model_graph = model.__repr__().split('\n')
        rnd_graph = rnd.__repr__().split('\n')


        description = {
            'comment': comment,
            'lr': lr,
            'lr_rnd': lr_rnd,
            'zeta': zeta,
            'beta': beta,
            'eps': eps,
            'gamma': gamma,
            'actor_critic_graph' : model_graph,
            'rnd_graph' : rnd_graph,
            #'actions': actions, 
            'date':  datetime.now().strftime("%d%m%Y_%H-%M-%S"),
            'old': old_fn
          }

        fn = os.path.join(self.save_description_path, 'description.yaml' )

        with io.open(fn, 'w', encoding='utf8') as outfile:
            yaml.dump(description, outfile, default_flow_style=False, allow_unicode=True)


    def update(self, action,
                    intrinsic_reward,
                    extrinsic_reward,
                    value_loss,
                    entropy_loss,
                    policy_loss):

        # Data
        self._actions.append(action)
        self._intrinsic_rewards.append(intrinsic_reward)
        self._extrinsic_rewards.append(extrinsic_reward)
        self._value_losses.append(value_loss)
        self._entropy_losses.append(entropy_loss)
        self._policy_losses.append(policy_loss)


    def consolidate(self, steps, episode, model, optimizer, rnd):
        self.episode = episode
        self.steps = steps
        # Data
        self._intrinsic_rewards_accummulative.append(np.sum(self._intrinsic_rewards))
        self._extrinsic_rewards_accummulative.append(np.sum(self._extrinsic_rewards))
        self._value_losses_accummulative.append(np.sum(self._value_losses))
        self._entropy_losses_accummulative.append(np.sum(self._entropy_losses))
        self._policy_losses_accummulative.append(np.sum(self._policy_losses))
        self._steps_accummulative.append(self.steps)
        

        # Checkpoints and Log
        if self.save_checkpoints and self.episode % self.checkpoint_every == 0 and self.episode != 0:
            fn = "e={}_ri={}_re={}_steps={}.pth".format(self.episode, 
                                                        np.sum(self._intrinsic_rewards), 
                                                        np.sum(self._extrinsic_rewards),
                                                        self.steps)

            checkpoint_path = os.path.join( self.save_model_path, fn)

            self.save_model(self.episode, model, optimizer, checkpoint_path )
            self.save_data(self.episode, model, rnd, fn)
        
        reward = np.sum(self._intrinsic_rewards) + np.sum(self._extrinsic_rewards)

        if self.save_best and reward >= self.best_reward:
            self.best_reward = reward
            if self.episode > 100:
                fn = "best_e={}_ri={}_re={}_steps={}.pth".format(self.episode, 
                                                        np.sum(self._intrinsic_rewards), 
                                                        np.sum(self._extrinsic_rewards),
                                                        self.steps)

                checkpoint_path = os.path.join( self.save_model_path, fn)

                self.save_model(self.episode, model, optimizer, checkpoint_path)
                self.save_data(self.episode, model, rnd, fn)

        # Flush
        self._actions = []
        self._intrinsic_rewards = []
        self._extrinsic_rewards = []
        self._value_losses = []
        self._entropy_losses = []
        self._policy_losses = []     

                
    def save_model(self, episode, model, optimizer, fn_model):
        if not fn_model.endswith(".pth"):
            fn_model += ".pth"

        checkpoint = {
                'episode': episode,
                'actor_critic_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }

        torch.save(checkpoint, fn_model)

    def save_data(self, episode, model, rnd, fn):
        self.tb.add_histogram('actions', torch.tensor(self._actions), episode)
        self.tb.add_histogram('policy_loss', torch.stack(self._policy_losses), episode)
        self.tb.add_histogram('entropy_loss', torch.stack(self._entropy_losses), episode)
        self.tb.add_histogram('value_loss', torch.stack(self._value_losses), episode)
        self.tb.add_histogram('intrinsic_reward', torch.stack(self._intrinsic_rewards), episode)
        self.tb.add_histogram('extrinsic_reward', torch.tensor(self._extrinsic_rewards), episode)
        for name, weight in model.named_parameters():
            self.tb.add_histogram(name, weight, episode)

        for name, weight in rnd.named_parameters():
            self.tb.add_histogram(name, weight, episode)

        for step in range(len(self._intrinsic_rewards)):
            self.tb.add_scalar('intrinsic_rewards_' + fn, self._intrinsic_rewards[step], step)

        for step in range(len(self._extrinsic_rewards)):
            self.tb.add_scalar('extrinsic_rewards_' + fn, self._extrinsic_rewards[step], step)

        for step in range(len(self._policy_losses)):
            self.tb.add_scalar('policy_losses_' + fn, self._policy_losses[step], step)

        for step in range(len(self._entropy_losses)):
            self.tb.add_scalar('entropy_losses_' + fn, self._entropy_losses[step], step)

        for step in range(len(self._value_losses)):
            self.tb.add_scalar('value_losses_' + fn, self._value_losses[step], step)


    def close(self):
        # Write in .csv
        df = pd.DataFrame({
            'intrinsic_rewards': self._intrinsic_rewards_accummulative,
            'extrinsic_rewards': self._extrinsic_rewards_accummulative,
            'value_losses': self._value_losses_accummulative,
            'entropy_losses': self._entropy_losses_accummulative,
            'policy_losses': self._policy_losses_accummulative,
            'steps': self._steps_accummulative
        })
        fn = os.path.join(self.save_data_path, 'data.csv')
        df.to_csv(fn)
        self.report(df)
        self.tb.close()


    def exception_arisen(self, episode, model, optimizer):
        fn = os.path.join(self.save_model_path, 'tmp_model')
        self.save_model(episode, model, optimizer, fn)
        self.close()

    def report(self, df):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        df.plot(subplots=True, ax=axes)

        fn = os.path.join(self.save_result_path, 'results')
        fig.savefig(fn)
        plt.close()


    def plot_reward(self, sliding_window=10, show=False, save=False, fn = "rewards.png" ):
        #rewards = self._moving_average(self._rewards, sliding_window)
        rewards = self._rewards
        plt.plot(range(len(rewards)), rewards, label= self.save_result_path )  

        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.save_result_path, fn ))

        if show:
            plt.show()
        plt.close()

    def plot_loss(self, sliding_window=10, show=False, save=False, fn = "losses.png" ):
        #rewards = self._moving_average(self._rewards, sliding_window)
        losses = self._losses
        plt.plot(range(len(losses)), losses, label= self.save_result_path )

        plt.xlabel("Episode")
        plt.ylabel("Total losses")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.save_result_path, fn))

        if show:
            plt.show()
        plt.close()

    
