from torch import nn
import torch
from src.utils import *

class SimplePointNet(nn.Module):
    def __init__(self, in_channels = 2, feature_num = 64):
        super(SimplePointNet, self).__init__()
        c1 = 64
        c2 = 128
        c3 = 256
        f1 = feature_num


        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels = c1,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv2 = nn.Conv1d(in_channels=c1, out_channels = c2,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv3 = nn.Conv1d(in_channels=c2, out_channels = c3,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(c3, f1)

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)
        x = self.fc1(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        sp = 256

        g1 = 16
        g2 = 32

        f1 = 512
        hh = 512
        f2 = 128
        f3 = 64

        f4 = 16
        #f5 = 64

        self.pointnet = SimplePointNet(feature_num = sp)

        self.odom_net = nn.Sequential(
                            nn.Linear(3, g1),                       nn.Sigmoid(),
                            nn.Linear(g1,g2),                        nn.ReLU(),
                            )

        self.net_prior = nn.Sequential(
                            nn.Linear(sp + g2, f1),                   nn.ReLU(),
                            )

        self.lstm = nn.LSTMCell(f1, hh)

        self.net_post = nn.Sequential(
                            nn.Linear(hh, f2),                        nn.ReLU(),
                            nn.Linear(f2, f3),                   nn.ReLU(),
                            )

        self.net_actor = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,action_dim),                nn.Softmax(0)
                            )

        self.net_critic_int = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,1)
                            )

        self.net_critic_ext = nn.Sequential(
                            nn.Linear(f3,f4),                        nn.ReLU(),
                            nn.Linear(f4,1)
                            )

    def forward(self, x):
        x, (hx, cx) = x
        o_odom  = x[0:3]
        o_lidar = x[3:]
        o_lidar = o_lidar.view(1,2,-1)

        obs = self.pointnet(o_lidar).squeeze()
        pose = self.odom_net(o_odom) 

        net_input = torch.cat( (obs, pose) )
        z = self.net_prior(net_input)


        z_e = z.view(-1,z.size(0))
        hx, cx = self.lstm(z_e, (hx, cx))

        z = hx.squeeze()
        z = self.net_post(z)

        return self.net_actor(z), self.net_critic_int(z), self.net_critic_ext(z), (hx, cx), net_input.detach()


class RND(nn.Module):
    def __init__(self, state_dim = 16, k = 8):
        super(RND, self).__init__()      
        self.first = True
        # f1 = state_dim
        # f2 = 256
        # f3 = 128
        # f4 = 64
        # f5 = 64


        # self.target =   nn.Sequential(
        #                     nn.Linear(f1, f2),                   nn.ReLU(),
        #                     nn.Linear(f2, f3),                   nn.ReLU(),
        #                     nn.Linear(f3, f4),                   nn.ReLU(),
        #                     nn.Linear(f4, f5),                   nn.ReLU(),
        #                     nn.Linear(f5, k),
        #                     )  

        # self.predictor = nn.Sequential(
        #                     nn.Linear(f1, f2),                   nn.ReLU(),
        #                     nn.Linear(f2, f3),                   nn.ReLU(),
        #                     nn.Linear(f3, f4),                   nn.ReLU(),
        #                     nn.Linear(f4, f5),                   nn.ReLU(),
        #                     nn.Linear(f5, k),
        #                     )  

        f1 = state_dim
        f2 = 16
        f3 = 8
        f4 = 8
        f5 = 64


        self.target =   nn.Sequential(
                            nn.Linear(f1, f2),                   nn.ReLU(),
                            nn.Linear(f2, f3),                   nn.ReLU(),
                            nn.Linear(f3, f4),                   nn.ReLU(),
                            nn.Linear(f4, k),                    nn.Softmax(0),
                            )  

        self.predictor = nn.Sequential(
                            nn.Linear(f1, f2),                   nn.ReLU(),
                            nn.Linear(f2, f3),                   nn.ReLU(),
                            nn.Linear(f3, f4),                   nn.ReLU(),
                            nn.Linear(f4, k),                   nn.Softmax(0),
                            )  


        self.predictor.apply(weights_init)
        self.target.apply(weights_init)

        for param in self.target.parameters():
            param.requires_grad = False
            

    def reset(self):
        self.predictor.apply(weights_init)
        self.target.apply(weights_init)

    def forward(self, x):
        to = self.target(x)
        po = self.predictor(x)

        mse = (to - po).pow(2).sum(0)

        int_reward =  mse.detach().float().unsqueeze(0)

        return to, po, int_reward

