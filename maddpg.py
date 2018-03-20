import common
from model import Actor
from model import Critic
from memory import Experience
import torch.optim as optim
from random_process import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import trange
import pandas as pd
from gym import wrappers


class MADDPG(object):
    def __init__(self, env, 
            mem_size=7*int(1e3), 
            lr_critic=1e-3, 
            lr_actor=1e-4, 
            epsilon=1., 
            max_epi=1500, 
            epsilon_decay=1./(1e5), 
            gamma=.99, 
            target_update_frequency=200, 
            batch_size=64,
            random_process=True,
            max_step=None
            ):
        self.CUDA = torch.cuda.is_available()

        self.orig_env = env #for recording
        if max_step is not None:
            self.orig_env._max_episode_steps = max_step
        self.env = self.orig_env
        self.N = 1
        if hasattr(self.env.unwrapped, 'N'):
            self.N = self.env.unwrapped.N
        self.N_S = self.env.observation_space.shape[0]
        self.N_A = self.env.action_space.shape[0]
        self.n_s = self.N_S/self.N
        self.n_a = self.N_A/self.N
        self.MAX_EPI = max_epi
        self.LOW = self.env.action_space.low
        self.HIGH = self.env.action_space.high
        
        self.actor = Actor(self.n_s, self.n_a)
        self.critic = Critic(self.N_S, self.N_A)
        self.target_actor = Actor(self.n_s, self.n_a)
        self.target_critic = Critic(self.N_S, self.N_A)
        self.target_actor.eval()
        self.target_critic.eval()
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        if self.CUDA:
            self.actor.cuda()
            self.critic.cuda()
            self.target_actor.cuda()
            self.target_critic.cuda()

        self.exp = Experience(mem_size)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=-lr_actor)
        self.random_processes = []
        for _ in xrange(self.N):
            random_process = OrnsteinUhlenbeckProcess(\
                    size=self.n_a, theta=.15, mu=0, sigma=.2)
            self.random_processes.append(random_process)
        self.EPSILON = epsilon 
        self.EPSILON_DECAY = epsilon_decay
        self.GAMMA = gamma
        self.TARGET_UPDATE_FREQUENCY = target_update_frequency
        self.BATCH_SIZE = batch_size

        title = {common.S_EPI:[], common.S_TOTAL_R:[]}
        self.data = pd.DataFrame(title)
        self.RAND_PROC = random_process
        
    def train(self, dir=None):
        if dir is not None:
            self.env = wrappers.Monitor(self.orig_env, '{}/train_record'.format(dir), force=True)
        update_counter = 0
        epsilon = self.EPSILON
        for epi in trange(self.MAX_EPI, desc='train epi', leave=True):
            for i in xrange(self.N):
                self.random_processes[i].reset_states()
            o = self.env.reset()
        
            counter = 0
            acc_r = 0
            while True:
                counter += 1
                
                if dir is not None:
                    self.env.render()
                
                a = np.zeros(self.N_A)
                for i in xrange(self.N):
                    tmp_o = o[i*self.n_s:(i+1)*self.n_s]
                    tmp_a = self.choose_action(tmp_o)
                    if self.RAND_PROC:
                        tmp_a += max(epsilon, 0)*self.random_processes[i].sample()
                        tmp_a = np.clip(tmp_a, -1., 1.)
                        epsilon -= self.EPSILON_DECAY
                    a[i*self.n_a:(i+1)*self.n_a] = tmp_a
                    
                o_, r, done, info = self.env.step(self.map_to_action(a))
                self.exp.push(o, a, r, o_)
                
                if epi>0: 
                    self.update_actor_critic()
                    update_counter += 1
                    if update_counter % self.TARGET_UPDATE_FREQUENCY == 0:
                        self.update_target()

                acc_r += r
                o = o_
                if done:
                    break
            s = pd.Series([epi, acc_r], index=[common.S_EPI, common.S_TOTAL_R])
            self.data = self.data.append(s, ignore_index=True)
    
    def choose_action(self, state):
        self.actor.eval()
        s = Variable(torch.Tensor(state)).unsqueeze(0)
        if self.CUDA:
            s = s.cuda()
        a = self.actor(s).data.cpu().numpy()[0].astype('float64')
        self.actor.train()
        return a

    def map_to_action(self, a):
        n = a.shape[0]
        return (self.LOW[:n]+self.HIGH[:n])/2 + a*(self.HIGH[:n]-self.LOW[:n])/2

    def update_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update_actor_critic(self):
        # sample minibatch
        minibatch = common.Transition(*zip(*self.exp.sample(self.BATCH_SIZE)))
        bat_o = Variable(torch.Tensor(minibatch.state))
        bat_a = Variable(torch.Tensor(minibatch.action))
        bat_r = Variable(torch.Tensor(minibatch.reward)).unsqueeze(1)
        bat_o_ = Variable(torch.Tensor(minibatch.next_state))
        if self.CUDA:
            bat_o = bat_o.cuda()
            bat_a = bat_a.cuda()
            bat_r = bat_r.cuda()
            bat_o_ = bat_o_.cuda()
        
        # update critic
        bat_a_o_ = Variable(bat_a.data.clone())
        for i in xrange(self.N):
            tmp_bat_o_ = bat_o_[:,i*self.n_s:(i+1)*self.n_s]
            tmp_bat_a_o_ = self.target_actor(tmp_bat_o_)
            bat_a_o_[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o_
        #bat_a_o_ = self.target_actor(bat_o_)

        Gt = bat_r + self.GAMMA * self.target_critic(bat_o_, bat_a_o_)
        Gt.detach_()
        eval_o = self.critic(bat_o, bat_a)
        criterion = nn.MSELoss()
        if self.CUDA:
            criterion.cuda()
        loss = criterion(eval_o, Gt)
        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()
        
        # update actor
        self.critic.eval()

        bat_a_o = Variable(bat_a.data.clone())
        for i in xrange(self.N):
            tmp_bat_o = bat_o[:,i*self.n_s:(i+1)*self.n_s]
            tmp_bat_a_o = self.actor(tmp_bat_o)
            bat_a_o[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o
        #bat_a_o = self.actor(bat_o)

        obj = torch.mean(self.critic(bat_o, bat_a_o))
        self.optim_actor.zero_grad()
        obj.backward()
        self.optim_actor.step()    
        self.critic.train()

    def test(self, dir=None, n=1):
        if dir is not None:
            self.env = wrappers.Monitor(self.orig_env, '{}/test_record'.format(dir), force=True, video_callable=lambda episode_id: True)

        title = {common.S_EPI:[], common.S_TOTAL_R:[]}
        df = pd.DataFrame(title)

        for epi in trange(n, desc='test epi', leave=True):
            o = self.env.reset()
            acc_r = 0
            while True:
                if dir is not None:
                    self.env.render()

                a = np.zeros(self.N_A)
                for i in xrange(self.N):
                    tmp_o = o[i*self.n_s:(i+1)*self.n_s]
                    tmp_a = self.choose_action(tmp_o)
                    a[i*self.n_a:(i+1)*self.n_a] = tmp_a

                o_, r, done, info = self.env.step(self.map_to_action(a))
                acc_r += r
                o = o_
                if done:
                    break
            s = pd.Series([epi, acc_r], index=[common.S_EPI, common.S_TOTAL_R])
            df = df.append(s, ignore_index=True)
        if dir is not None:
            df.to_csv('{}/test_data.csv'.format(dir))
        else:
            print df

    def save(self, dir):
        torch.save(self.actor.state_dict(), '{}/actor.pt'.format(dir))
        torch.save(self.critic.state_dict(), '{}/critic.pt'.format(dir))
        self.data.to_csv('{}/train_data.csv'.format(dir))

    def load_actor(self, dir):
        self.actor.load_state_dict(torch.load(dir))

    def load_critic(self, dir):
        self.critic.load_state_dict(torch.load(dir))

    def get_data(self):
        return self.data

    

        


