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
        
        f1 = 64
        f2 = 32
        f3 = 32

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
        
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        f4 = 16

        self.local_notion = LocalNotion(state_dim - 2, f4)

        g1 = 32
        g2 = 16

        self.net = nn.Sequential(
                            nn.Linear(f4 + 2, g1),                   nn.ReLU(),
                            nn.Linear(g1,g2),                        nn.ReLU(),
                            nn.Linear(g2,action_dim),                nn.Softmax(0)
                            )
    def forward(self, x):
        o_odom  = x[0:2]
        o_lidar = x[2:]
        out = self.local_notion(o_lidar)
        net_input = torch.cat( (out, o_odom) )
        return self.net(net_input)



class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        f4 = 16

        self.local_notion = LocalNotion(state_dim - 2, f4)

        g1 = 32
        g2 = 16

        self.net = nn.Sequential(
                            nn.Linear(f4 + 2, g1),                   nn.ReLU(),
                            nn.Linear(g1,g2),                        nn.ReLU(),
                            nn.Linear(g2,1),    
                            )
    def forward(self, x):
        o_odom  = x[0:2]
        o_lidar = x[2:]
        out = self.local_notion(o_lidar)
        net_input = torch.cat( (out, o_odom) )
        return self.net(net_input)