from torch import nn
import torch
from src.utils import *

class SimplePointNet(nn.Module):
    def __init__(self, in_channels = 2, feature_num = 64):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels = 32,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels = 64,
                              kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels = feature_num,
                              kernel_size=1, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(2)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        f4 = 32
        self.pointnet = SimplePointNet(feature_num = f4)

        g1 = 32
        g2 = 16
        g3 = 8
        hh = 16

        self.net = nn.Sequential(
                            nn.Linear(f4 + 3, g1),                   nn.ReLU(),
                            nn.Linear(g1,g2),                        nn.ReLU(),
                            )

        self.lstm = nn.LSTMCell(g2, hh)

        self.net_actor = nn.Sequential(
                            nn.Linear(hh,g3),                        nn.ReLU(),
                            nn.Linear(g3,action_dim),                nn.Softmax(0)
                            )

        self.net_critic_int = nn.Sequential(
                            nn.Linear(hh,g3),                        nn.ReLU(),
                            nn.Linear(g3,1)
                            )

        self.net_critic_ext = nn.Sequential(
                            nn.Linear(hh,g3),                        nn.ReLU(),
                            nn.Linear(g3,1)
                            )

    def forward(self, x):
        x, (hx, cx) = x
        o_odom  = x[0:3]
        o_lidar = x[3:]
        o_lidar = o_lidar.view(1,2,-1)

        h = self.pointnet(o_lidar).squeeze()
        net_input = torch.cat( (h, o_odom) )
        net_out = self.net(net_input)
        net_out_ex = net_out.view(-1,net_out.size(0))
        hx, cx = self.lstm(net_out_ex, (hx, cx))
        
        return self.net_actor(hx), self.net_critic_int(hx), self.net_critic_ext(hx), (hx, cx), net_out


class RND(nn.Module):
    def __init__(self, state_dim = 16, k = 16):
        super(RND, self).__init__()      
        f1 = state_dim
        f2 = 32
        f3 = 16
        self.target =   nn.Sequential(
                            nn.Linear(f1, f3),                   nn.Sigmoid(),
                            #nn.Linear(f2, f3),                   nn.ReLU(),
                            nn.Linear(f3, k),                    nn.Sigmoid()
                            )  
        self.predictor = nn.Sequential(
                            nn.Linear(f1, f3),                   nn.Sigmoid(),
                            #nn.Linear(f2, f3),                   nn.ReLU(),
                            nn.Linear(f3, k),                    nn.Sigmoid()
                            )


        self.predictor.apply(weights_init)
        self.target.apply(weights_init)

        for param in self.target.parameters():
            param.requires_grad = False

        self.input_norm = RunningMeanStd(shape = state_dim)
        self.output_norm = RunningMeanStdOne() #RunningFixed()

    def reset(self):
        self.predictor.apply(weights_init)
        self.target.apply(weights_init)
        #self.output_norm.reset()

    def forward(self, xn):
        #xn = self.input_norm.update(x)
        to = self.target(xn).detach()
        po = self.predictor(xn)
        mse = (to - po).pow(2).sum(0) / 2
        int_reward = self.output_norm.update(mse.detach().float().unsqueeze(0))
        return to, po, int_reward


