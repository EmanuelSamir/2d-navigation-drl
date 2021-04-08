from torch import nn
import torch


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.input_dim = 2
        self.layer1 = 2
        self.layer2 = 4

        self.net = nn.Sequential(
                    nn.Linear(self.input_dim, self.layer1),                       nn.ReLU(),
                    nn.Linear(self.layer1, self.layer2),                          nn.ReLU(),
                    )

    def forward(self, x):
        return self.net(x)

class LocalNotion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LocalNotion, self).__init__()
        
        self.blocks = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        f1 = 32
        f2 = 16
        f3 = 16

        self.device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for _ in range(self.input_dim//2):
            block = Block().to(device = self.device)
            self.blocks.append(block)
        
        self.net = nn.Sequential(
                                nn.Linear(block.layer2 * self.input_dim//2, f1),               nn.ReLU(),
                                nn.Linear(f1, f2),                       nn.ReLU(),
                                nn.Linear(f2, f3),                       nn.ReLU(),
                                nn.Linear(f3, self.output_dim),                       nn.ReLU(),
        )

    def forward(self, x):
        out = []
        for b in range(self.input_dim//2):
            z = self.blocks[b](x[2*b:2*b + 2])
            out.append(z)
            
        z = torch.cat(out)
        z = self.net(z)
            
        return z
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        f4 = 16

        self.local_notion = LocalNotion(state_dim - 3, f4)

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

        self.net_critic = nn.Sequential(
                            nn.Linear(hh,g3),                        nn.ReLU(),
                            nn.Linear(g3,1)
                            )


    def forward(self, x):
        x, (hx, cx) = x
        o_odom  = x[0:3]
        o_lidar = x[3:]
        h = self.local_notion(o_lidar)
        net_input = torch.cat( (h, o_odom) )
        net_out = self.net(net_input)
        net_out = net_out.view(-1,net_out.size(0))
        hx, cx = self.lstm(net_out, (hx, cx))
        
        return self.net_actor(hx), self.net_critic(hx), (hx, cx)
