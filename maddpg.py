import common
from model import Actor
from model import Critic
#from memory import Experience
from per import Experience #prioritized experience replay
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
import os
from normalized_env import NormalizedEnv
import math
import copy


class MADDPG(object):
    def __init__(self, env, 
            mem_size=int(1e6), 
            lr_critic=1e-3, 
            lr_actor=1e-4, 
            epsilon=1., 
            max_epi=int(1e4), 
            epsilon_decay=1./(1e5), 
            gamma=.99, 
            target_update_frequency=200, 
            batch_size=64,
            random_process=True,
            max_step=None,
            dynamic_actor_update=False,
            ):
        self.CUDA = torch.cuda.is_available()
        self.DYNAMIC_ACTOR_UPDATE = dynamic_actor_update
        self.ENV_NORMALIZED = env.class_name() == 'NormalizedEnv'

        self.orig_env = (env) #for recording
        if max_step is not None:
            tmp_env = env
            if isinstance(tmp_env, gym.Wrapper):
                while(tmp_env.class_name() != 'TimeLimit'):
                    tmp_env = tmp_env.env
                tmp_env._max_episode_steps = max_step
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

        # pop-art
        self.update_counter = 0
        self.beta = .1
        self.y_mean = 0.
        self.y_square_mean = 0.
        self.target_y_mean = self.y_mean
        self.target_y_square_mean = self.y_square_mean

        # per
        self.total_step = 0
        self.PARTITION_NUM = 100
        self.LEARN_START = mem_size/self.PARTITION_NUM+1
        exp_conf = {
                'size': mem_size,
                'learn_start': self.LEARN_START,
                'partition_num': self.PARTITION_NUM,
                'total_step': self.MAX_EPI * 50,
                'batch_size': batch_size,
                }
        self.exp = Experience(exp_conf)

        # uniform er
        #self.exp = Experience(mem_size)
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
        
    def train(self, dir=None, interval=1000):
        if dir is not None:
            self.env = wrappers.Monitor(self.orig_env, '{}/train_record'.format(dir), force=True)
            os.mkdir(os.path.join(dir, 'models'))
        self.update_counter = 0
        self.total_step = 0
        epsilon = self.EPSILON
        for epi in trange(self.MAX_EPI, desc='train epi', leave=True):
            for i in xrange(self.N):
                self.random_processes[i].reset_states()
            o = self.env.reset()
        
            counter = 0
            acc_r = 0
            while True:
                counter += 1
                self.total_step += 1
                
                #if dir is not None:
                #    self.env.render()
                
                a = np.zeros(self.N_A)
                for i in xrange(self.N):
                    tmp_o = o[i*self.n_s:(i+1)*self.n_s]
                    tmp_a = self.choose_action(tmp_o)
                    if self.RAND_PROC:
                        tmp_a += max(epsilon, 0)*self.random_processes[i].sample()
                        tmp_a = np.clip(tmp_a, -1., 1.)
                        epsilon -= self.EPSILON_DECAY
                    a[i*self.n_a:(i+1)*self.n_a] = tmp_a
                    
                if self.ENV_NORMALIZED:
                    o_, r, done, info = self.env.step(a)
                else:
                    o_, r, done, info = self.env.step(self.map_to_action(a))

                # per
                if self.DYNAMIC_ACTOR_UPDATE:
                    self.exp.store(common.Transition_agent(o, a, r, o_, done, info['agent']))
                else:
                    self.exp.store(common.Transition(o, a, r, o_, done))

                # uer
                #self.exp.push(o, a, r, o_, done)
                
                #if epi>0: 
                #if len(self.exp.mem)>self.BATCH_SIZE*25: 
                if self.total_step > self.LEARN_START:
                    self.update_counter += 1
                    self.update_actor_critic()
                    if self.update_counter % self.TARGET_UPDATE_FREQUENCY == 0:
                        self.update_target()

                acc_r += r
                o = o_
                if done:
                    break
            if dir is not None:
                if (epi+1) % interval == 0:
                    self.save(os.path.join(dir, 'models'), str(epi+1), save_data=False)
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
        # pop-art
        self.target_y_mean = self.y_mean
        self.target_y_square_mean = self.y_square_mean

    def update_actor_critic(self):
        # sample minibatch

        # per
        minibatch, w, e_id = self.exp.sample(self.total_step)
        if self.DYNAMIC_ACTOR_UPDATE:
            minibatch = common.Transition_agent(*zip(*minibatch))
        else:
            minibatch = common.Transition(*zip(*minibatch))

        #minibatch = common.Transition(*zip(*self.exp.sample(self.BATCH_SIZE)))
        bat_o = Variable(torch.Tensor(minibatch.state))
        bat_a = Variable(torch.Tensor(minibatch.action))
        bat_r = Variable(torch.Tensor(minibatch.reward)).unsqueeze(1)
        bat_o_ = Variable(torch.Tensor(minibatch.next_state))
        bat_not_done_mask = list(map(lambda done: 0 if done else 1, minibatch.done))
        bat_not_done_mask = Variable(torch.ByteTensor(bat_not_done_mask)).unsqueeze(1)
        if self.CUDA:
            bat_o = bat_o.cuda()
            bat_a = bat_a.cuda()
            bat_r = bat_r.cuda()
            bat_o_ = bat_o_.cuda()
            bat_not_done_mask = bat_not_done_mask.cuda()
        bat_agent = minibatch.agent if self.DYNAMIC_ACTOR_UPDATE else None
        
        # update critic
        bat_a_o_ = Variable(bat_a.data.clone())
        for i in xrange(self.N):
            tmp_bat_o_ = bat_o_[:,i*self.n_s:(i+1)*self.n_s]
            tmp_bat_a_o_ = self.target_actor(tmp_bat_o_)
            bat_a_o_[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o_
        #bat_a_o_ = self.target_actor(bat_o_)

        Gt = bat_r
        #Gt[bat_not_done_mask] += self.GAMMA * self.target_critic(bat_o_, bat_a_o_)[bat_not_done_mask]

        # pop-art
        target_y_delta = math.sqrt(self.target_y_square_mean - self.target_y_mean**2)
        Gt[bat_not_done_mask] += self.GAMMA * (target_y_delta * self.target_critic(bat_o_, bat_a_o_)[bat_not_done_mask] + self.target_y_mean)

        beta_t = self.beta * 1. / (1 - math.pow(1-self.beta, self.update_counter))
        y_t = torch.mean(Gt).data.cpu().numpy()[0]
        y_square_t = torch.mean(Gt**2).data.cpu().numpy()[0]
        y_mean_new = (1.-beta_t) * self.y_mean + beta_t * y_t
        y_square_mean_new = (1.-beta_t) * self.y_square_mean + beta_t * y_square_t
        y_delta = math.sqrt(self.y_square_mean - self.y_mean**2)
        y_delta_new = math.sqrt(y_square_mean_new - y_mean_new**2)

        self.critic.fc_final.weight.data *= y_delta / y_delta_new
        self.critic.fc_final.bias.data *= y_delta
        self.critic.fc_final.bias.data += (self.y_mean - y_mean_new)
        self.critic.fc_final.bias.data /= y_delta_new

        self.y_mean = y_mean_new
        self.y_square_mean = y_square_mean_new
        y_delta = y_delta_new

        #Gt -= self.y_mean
        #Gt /= y_delta

        Gt.detach_()
        eval_o = y_delta * self.critic(bat_o, bat_a) + self.y_mean

        # per
        w = Variable(torch.Tensor(w)).unsqueeze(1)
        if self.CUDA:
            w = w.cuda()
        loss = (eval_o - Gt)**2
        delta = loss.data.cpu().numpy().copy() if self.CUDA else loss.data.numpy().copy() 
        self.exp.update_priority(e_id, delta)
        loss = w*loss
        loss = torch.mean(loss)

        # uer
        #criterion = nn.MSELoss()
        #if self.CUDA:
        #    criterion.cuda()
        #loss = criterion(eval_o, Gt)

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()
        
        # update actor
        self.critic.eval()

        bat_a_o = Variable(bat_a.data.clone())

        # only update relevant agents
        if self.DYNAMIC_ACTOR_UPDATE:
            # group data by agent type, so as to modify data in batch mode
            # agent_indexes[i]: list of sample indexes related to agent i
            agent_indexes = []
            for _ in range(self.N):
                agent_indexes.append([])
            for index, agents in enumerate(bat_agent):
                if len(agents)==0:
                    for agent in range(self.N):
                        agent_indexes[agent].append(index)
                else:
                    for agent in agents:
                        agent_indexes[agent].append(index)
            # to avoid using syntax like bat_a_o[index, xxx], use torch.cat instead
            bat_a_o = []
            for i in xrange(self.N):
                index = agent_indexes[i]
                tmp_bat_o = bat_o[index,:][:,i*self.n_s:(i+1)*self.n_s]
                tmp_bat_a_o = self.actor(tmp_bat_o)
                curr_a_o = Variable(bat_a[:,i*self.n_a:(i+1)*self.n_a].data.clone())
                if len(curr_a_o.data.shape)==1:
                    curr_a_o = curr_a_o.unsqueeze(1)
                curr_a_o[index,:] = tmp_bat_a_o
                bat_a_o.append(curr_a_o)
            bat_a_o = torch.cat(bat_a_o, 1)
        else:
            for i in xrange(self.N):
                tmp_bat_o = bat_o[:,i*self.n_s:(i+1)*self.n_s]
                tmp_bat_a_o = self.actor(tmp_bat_o)
                bat_a_o[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o
        #bat_a_o = self.actor(bat_o)

        obj = torch.mean(y_delta * self.critic(bat_o, bat_a_o) + self.y_mean)
        # clear the grad from previous critic update
        self.optim_critic.zero_grad()
        self.optim_actor.zero_grad()
        obj.backward()
        print self.actor.fc1.weight.grad

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
                #if dir is not None:
                #    self.env.render()

                a = np.zeros(self.N_A)
                for i in xrange(self.N):
                    tmp_o = o[i*self.n_s:(i+1)*self.n_s]
                    tmp_a = self.choose_action(tmp_o)
                    a[i*self.n_a:(i+1)*self.n_a] = tmp_a

                if self.ENV_NORMALIZED:
                    o_, r, done, info = self.env.step(a)
                else:
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

    def save(self, dir, suffix='', save_data=True):
        torch.save(self.actor.state_dict(), '{}/actor{}.pt'.format(dir, suffix))
        torch.save(self.critic.state_dict(), '{}/critic{}.pt'.format(dir, suffix))
        if save_data:
            self.data.to_csv('{}/train_data{}.csv'.format(dir, suffix))
        # MAYBE have to save and load y_mean and y_square_mean if using pop-art


    def load_actor(self, dir):
        self.actor.load_state_dict(torch.load(dir))

    def load_critic(self, dir):
        self.critic.load_state_dict(torch.load(dir))

    def get_data(self):
        return self.data

    

        


