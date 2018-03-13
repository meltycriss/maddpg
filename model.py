import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

uf_lb = -3.*(1e-3)
uf_ub = 3.*(1e-3)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic, self).__init__()
        self.fcS = nn.Linear(n_s, 400)
        #nn.init.kaiming_normal(self.fcS.weight)
        #nn.init.uniform(self.fcS.weight, uf_lb, uf_ub)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(400)
        
        self.fcA = nn.Linear(n_a, 400)
        #nn.init.kaiming_normal(self.fcA.weight)
        #nn.init.uniform(self.fcA.weight, uf_lb, uf_ub)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(400)
        
        self.fc1 = nn.Linear(400, 300)
        #nn.init.kaiming_normal(self.fc1.weight)
        #nn.init.uniform(self.fc1.weight, uf_lb, uf_ub)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(300)
        
        self.fc2 = nn.Linear(300, 1)
        #nn.init.kaiming_normal(self.fc2.weight)
        nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
#         s = F.relu(self.bnS(self.fcS(x)))
#         a = F.relu(self.bnA(self.fcA(y)))
#         o = F.relu(self.bn1(self.fc1(s+a)))
#         o = self.fc2(o)
               
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = self.bn1(self.fc1(o))
        o = F.relu(o)
        o = (self.fc2(o))
        
        return o

class Actor(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_s, 400)
        #nn.init.kaiming_normal(self.fc1.weight)
        #nn.init.uniform(self.fc1.weight, uf_lb, uf_ub)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(400)
        
        self.fc2 = nn.Linear(400, 300)
        #nn.init.kaiming_normal(self.fc2.weight)
        #nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(300)
        
        self.fc3 = nn.Linear(300, n_a)
        #nn.init.kaiming_normal(self.fc3.weight)
        nn.init.uniform(self.fc3.weight, uf_lb, uf_ub)
        
    def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)
        
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = (self.fc3(x))
        x = F.tanh(x)
        return x
