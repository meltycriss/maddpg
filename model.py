import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

CriticRegistry = {}
ActorRegistry = {}

uf_lb = -3.*(1e-3)
uf_ub = 3.*(1e-3)

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

#########################################################################
# Critic
#########################################################################

#######################################################
# width
#######################################################

# standard critic
class Critic(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic, self).__init__()
        self.fcS = nn.Linear(n_s, 400)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(400)
        
        self.fc1 = nn.Linear(400+n_a, 300)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(300)
        
        self.fc2 = nn.Linear(300, 1)
        nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
    
    def forward(self, s, a):
        o = F.relu(self.bnS(self.fcS(s)))
        o = torch.cat([o, a], 1)
        o = F.relu(self.bn1(self.fc1(o)))
        o = self.fc2(o)
        return o
CriticRegistry['standard'] = Critic

class Critic_52c(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_52c, self).__init__()
        self.fcS = nn.Linear(n_s, 512)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512+n_a, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 1)
        nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
    
    def forward(self, s, a):
        o = F.relu(self.bnS(self.fcS(s)))
        o = torch.cat([o, a], 1)
        o = F.relu(self.bn1(self.fc1(o)))
        o = self.fc2(o)
        return o
CriticRegistry['52c'] = Critic_52c

class Critic_21c(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_21c, self).__init__()
        self.fcS = nn.Linear(n_s, 256)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(256)
        
        self.fc1 = nn.Linear(256+n_a, 128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 1)
        nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
    
    def forward(self, s, a):
        o = F.relu(self.bnS(self.fcS(s)))
        o = torch.cat([o, a], 1)
        o = F.relu(self.bn1(self.fc1(o)))
        o = self.fc2(o)
        return o
CriticRegistry['21c'] = Critic_21c

class Critic_16c(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_16c, self).__init__()
        self.fcS = nn.Linear(n_s, 128)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128+n_a, 64)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 1)
        nn.init.uniform(self.fc2.weight, uf_lb, uf_ub)
    
    def forward(self, s, a):
        o = F.relu(self.bnS(self.fcS(s)))
        o = torch.cat([o, a], 1)
        o = F.relu(self.bn1(self.fc1(o)))
        o = self.fc2(o)
        return o
CriticRegistry['16c'] = Critic_16c

#######################################################
# embedding
#######################################################
class Critic_43(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_43, self).__init__()
        self.fcS = nn.Linear(n_s, 400)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(400)
        
        self.fcA = nn.Linear(n_a, 400)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(400)
        
        self.fc1 = nn.Linear(400, 300)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(300)
        
        self.fc_final = nn.Linear(300, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = self.bn1(self.fc1(o))
        o = F.relu(o)
        o = (self.fc_final(o))
        return o
CriticRegistry['43'] = Critic_43

class Critic_52(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_52, self).__init__()
        self.fcS = nn.Linear(n_s, 512)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(512)
        
        self.fcA = nn.Linear(n_a, 512)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc_final = nn.Linear(256, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = F.relu(self.bn1(self.fc1(o)))
        
        o = (self.fc_final(o))
        return o
CriticRegistry['52'] = Critic_52


class Critic_21(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_21, self).__init__()
        self.fcS = nn.Linear(n_s, 256)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(256)
        
        self.fcA = nn.Linear(n_a, 256)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(256)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc_final = nn.Linear(128, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = self.bn1(self.fc1(o))
        o = F.relu(o)
        o = (self.fc_final(o))
        return o
CriticRegistry['21'] = Critic_21

class Critic_16(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_16, self).__init__()
        self.fcS = nn.Linear(n_s, 128)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(128)
        
        self.fcA = nn.Linear(n_a, 128)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc_final = nn.Linear(64, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = self.bn1(self.fc1(o))
        o = F.relu(o)
        o = (self.fc_final(o))
        return o
CriticRegistry['16'] = Critic_16

#######################################################
# depth
#######################################################
class Critic_5216(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_5216, self).__init__()
        self.fcS = nn.Linear(n_s, 512)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(512)
        
        self.fcA = nn.Linear(n_a, 512)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc_final = nn.Linear(64, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = F.relu(self.bn1(self.fc1(o)))
        o = F.relu(self.bn2(self.fc2(o)))
        o = F.relu(self.bn3(self.fc3(o)))
        
        o = (self.fc_final(o))
        return o
CriticRegistry['5216'] = Critic_5216

class Critic_521(nn.Module):
    def __init__(self, n_s, n_a):
        super(Critic_521, self).__init__()
        self.fcS = nn.Linear(n_s, 512)
        self.fcS.weight.data = fanin_init(self.fcS.weight.data.size())
        self.bnS = nn.BatchNorm1d(512)
        
        self.fcA = nn.Linear(n_a, 512)
        self.fcA.weight.data = fanin_init(self.fcA.weight.data.size())
        self.bnA = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc_final = nn.Linear(128, 1)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
    
    def forward(self, x, y):
        s = self.bnS(self.fcS(x))
        a = self.bnA(self.fcA(y))
        o = F.relu(s+a)
        o = F.relu(self.bn1(self.fc1(o)))
        o = F.relu(self.bn2(self.fc2(o)))
        
        o = (self.fc_final(o))
        return o
CriticRegistry['521'] = Critic_521

#########################################################################
# Actor
#########################################################################

# standard actor
class Actor(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_s, 400)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(400)
        
        self.fc2 = nn.Linear(400, 300)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(300)
        
        self.fc3 = nn.Linear(300, n_a)
        nn.init.uniform(self.fc3.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = (self.fc3(x))
        x = F.tanh(x)
        return x
ActorRegistry['standard'] = Actor

#######################################################
# width
#######################################################
class Actor_52(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor_52, self).__init__()
        self.fc1 = nn.Linear(n_s, 512)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, n_a)
        nn.init.uniform(self.fc3.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = (self.fc3(x))
        x = F.tanh(x)
        return x
ActorRegistry['52'] = Actor_52

class Actor_21(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor_21, self).__init__()
        self.fc1 = nn.Linear(n_s, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, n_a)
        nn.init.uniform(self.fc3.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = (self.fc3(x))
        x = F.tanh(x)
        return x
ActorRegistry['21'] = Actor_21

class Actor_16(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor_16, self).__init__()
        self.fc1 = nn.Linear(n_s, 128)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, n_a)
        nn.init.uniform(self.fc3.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = (self.fc3(x))
        x = F.tanh(x)
        return x
ActorRegistry['16'] = Actor_16

#######################################################
# depth
#######################################################
class Actor_5216(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor_5216, self).__init__()
        self.fc1 = nn.Linear(n_s, 512)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 64)
        self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())
        self.bn4 = nn.BatchNorm1d(64)
        
        self.fc_final = nn.Linear(64, n_a)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))

        x = (self.fc_final(x))
        x = F.tanh(x)
        return x
ActorRegistry['5216'] = Actor_5216

class Actor_521(nn.Module):
    def __init__(self, n_s, n_a):
        super(Actor_521, self).__init__()
        self.fc1 = nn.Linear(n_s, 512)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc_final = nn.Linear(128, n_a)
        nn.init.uniform(self.fc_final.weight, uf_lb, uf_ub)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        x = (self.fc_final(x))
        x = F.tanh(x)
        return x
ActorRegistry['521'] = Actor_521


