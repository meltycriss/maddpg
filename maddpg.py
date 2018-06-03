import common
from model import ActorRegistry
from model import CriticRegistry
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
import inspect
import pickle
import random
from logger import Logger
import util


class MADDPG(object):
    def __init__(self, env, 
            mem_size=int(1e6), 
            lr_critic=1e-3, 
            lr_actor=1e-4, 
            max_epi=int(1e4), 
            epsilon_decay=1./(1e5), 
            gamma=.99, 
            target_update_frequency=200, 
            batch_size=64,
            random_process_mode='default',
            max_step=None,
            actor_update_mode='default',
            popart=False,
            actor='standard',
            critic='43',
            epsilon_start=1.,
            epsilon_end=.01,
            epsilon_rate=1./200,
            partition_num=100,
            env_log_freq=100,
            model_log_freq=500,
            target_update_mode='hard',
            tau=1e-3,
            grad_clip_mode=None,
            grad_clip_norm=5.,
            critic_weight_decay=0.,
            ):
        # configuration log
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.config = ['{}: {}'.format(arg, values[arg]) for arg in args]

        self.CUDA = torch.cuda.is_available()
        self.ENV_NORMALIZED = env.class_name() == 'NormalizedEnv'
        self.POPART = popart
        self.actor_update_mode=actor_update_mode

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
        
        self.actor = ActorRegistry[actor](self.n_s, self.n_a)
        self.critic = CriticRegistry[critic](self.N_S, self.N_A)
        self.target_actor = ActorRegistry[actor](self.n_s, self.n_a)
        self.target_critic = CriticRegistry[critic](self.N_S, self.N_A)
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
        self.PARTITION_NUM = partition_num
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
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=critic_weight_decay)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=-lr_actor)
        self.random_processes = []
        for _ in xrange(self.N):
            random_process = OrnsteinUhlenbeckProcess(\
                    size=self.n_a, theta=.15, mu=0, sigma=.2)
            self.random_processes.append(random_process)
        self.EPSILON_START = epsilon_start
        self.EPSILON_END = epsilon_end
        # only default random process mode will use epsilon decay
        self.EPSILON_DECAY = epsilon_decay
        # other random process mode will use epsilon rate
        self.EPSILON_RATE = epsilon_rate
        self.GAMMA = gamma
        self.TARGET_UPDATE_FREQUENCY = target_update_frequency
        self.BATCH_SIZE = batch_size
        self.target_update_mode = target_update_mode
        self.tau = tau

        title = {common.S_EPI:[], common.S_TOTAL_R:[]}
        self.data = pd.DataFrame(title)
        self.RAND_PROC = random_process_mode

        self.grad_clip_mode = grad_clip_mode
        self.grad_clip_norm = grad_clip_norm

        # logger
        self.logger = None
        self.env_log_freq = env_log_freq
        self.model_log_freq = model_log_freq
        self.step = 0
        
    def train(self, dir=None, interval=1000):
        if dir is not None:
            self.env = wrappers.Monitor(self.orig_env, '{}/train_record'.format(dir), force=True)
            os.mkdir(os.path.join(dir, 'models'))
            self.logger = Logger('{}/logs'.format(dir))
        self.update_counter = 0
        self.total_step = 0
        epsilon = self.EPSILON_START
        self.update_target()
        for self.epi in trange(self.MAX_EPI, desc='train epi', leave=True):
            for i in xrange(self.N):
                self.random_processes[i].reset_states()
            o = self.env.reset()
        
            self.step = 0
            acc_r = 0
            # log
            epi_log = {}
            noise_ratios = []
            while True:
                self.step += 1
                self.total_step += 1
                
                #if dir is not None:
                #    self.env.render()
                
                if 'chunk' in self.RAND_PROC:
                    noise_flag = True if random.random()<epsilon else False
                    epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) *\
                            math.exp(-1. * self.total_step * self.EPSILON_RATE)
                a = np.zeros(self.N_A)
                for i in xrange(self.N):
                    tmp_o = o[i*self.n_s:(i+1)*self.n_s]
                    tmp_a = self.choose_action(tmp_o)

                    # different noise mode
                    # decaying noise
                    if 'default' in self.RAND_PROC:
                        noise = max(epsilon, 0)*self.random_processes[i].sample()
                        tmp_a += noise
                        epsilon -= self.EPSILON_DECAY
                    if 'exp' in self.RAND_PROC:
                        noise = max(epsilon, 0)*self.random_processes[i].sample()
                        tmp_a += noise
                        epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) *\
                                math.exp(-1. * self.total_step * self.EPSILON_RATE)
                    # epsilon greedy like noise
                    if 'sep' in self.RAND_PROC:
                        noise = self.random_processes[i].sample() if random.random()<epsilon else 0.
                        tmp_a += noise
                        epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) *\
                                math.exp(-1. * self.total_step * self.EPSILON_RATE)
                    if 'chunk' in self.RAND_PROC:
                        noise = self.random_processes[i].sample() if noise_flag else 0.
                        tmp_a += noise

                    tmp_a = np.clip(tmp_a, -1., 1.)
                    a[i*self.n_a:(i+1)*self.n_a] = tmp_a

                    # log
                    if (self.epi+1) % self.env_log_freq == 0:
                        noise_ratios.append(np.linalg.norm(noise) / np.linalg.norm(tmp_a))
#                    min_noise_ratio = noise_ratio if self.step==1 and i==0 else min(min_noise_ratio, noise_ratio)
#                    max_noise_ratio = noise_ratio if self.step==1 and i==0 else max(min_noise_ratio, noise_ratio)
#                    avg_noise_ratio = noise_ratio if self.step==1 and i==0 else avg_noise_ratio + 1./self.step*(noise_ratio-avg_noise_ratio)
                    
                if self.ENV_NORMALIZED:
                    o_, r, done, info = self.env.step(a)
                else:
                    o_, r, done, info = self.env.step(self.map_to_action(a))

                # per
                if self.actor_update_mode=='dynamic':
                    self.exp.store(common.Transition_agent(o, a, r, o_, done, info['agent']))
                else:
                    self.exp.store(common.Transition(o, a, r, o_, done))

                # uer
                #self.exp.push(o, a, r, o_, done)
                
                #if self.epi>0: 
                #if len(self.exp.mem)>self.BATCH_SIZE*25: 
                if self.total_step > self.LEARN_START:
                    self.update_counter += 1
                    self.update_actor_critic()
                    if self.update_counter % self.TARGET_UPDATE_FREQUENCY == 0:
                        self.update_target()

                acc_r += r
                o = o_

                # epi_log
                if (self.epi+1) % self.env_log_freq == 0:
                    if info.has_key('log_info'):
                        log_info = info['log_info']
                        if log_info['int_coll']:
                            epi_log['end_status'] = 0
                        elif log_info['ext_coll']:
                            epi_log['end_status'] = 1
                        elif log_info['success']:
                            epi_log['end_status'] = 2
                        else:
                            epi_log['end_status'] = 3
                        epi_log['min_goal_dis'] = min(log_info['goal_dis'], epi_log['min_goal_dis']) if epi_log.has_key('min_goal_dis') else log_info['goal_dis']
                        epi_log['avg_avg_agent_center_dis'] = epi_log['avg_avg_agent_center_dis'] + 1./self.step * (log_info['avg_agent_center_dis']-epi_log['avg_avg_agent_center_dis']) if epi_log.has_key('avg_avg_agent_center_dis') else log_info['avg_agent_center_dis']
                        epi_log['min_avg_agent_center_dis'] = min(epi_log['min_avg_agent_center_dis'], log_info['avg_agent_center_dis']) if epi_log.has_key('min_avg_agent_center_dis') else log_info['avg_agent_center_dis']
                        epi_log['avg_min_inter_agent_dis'] = epi_log['avg_min_inter_agent_dis'] + 1./self.step * (log_info['min_inter_agent_dis']-epi_log['avg_min_inter_agent_dis']) if epi_log.has_key('avg_min_inter_agent_dis') else log_info['min_inter_agent_dis']
                        epi_log['min_min_inter_agent_dis'] = min(epi_log['min_min_inter_agent_dis'], log_info['min_inter_agent_dis']) if epi_log.has_key('min_min_inter_agent_dis') else log_info['min_inter_agent_dis']

                if done:
                    break
            if dir is not None:
                if (self.epi+1) % interval == 0:
                    self.save(os.path.join(dir, 'models'), str(self.epi+1), save_data=False)
            s = pd.Series([self.epi, acc_r], index=[common.S_EPI, common.S_TOTAL_R])
            self.data = self.data.append(s, ignore_index=True)
            
            # log
            if (self.epi+1) % self.env_log_freq == 0:
                for key, value in epi_log.items():
                    self.logger.scalar_summary(key, value, self.epi+1)
                self.logger.histo_summary('noise_ratio', np.array(noise_ratios), self.epi+1)
    
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
        if 'hard' in self.target_update_mode:
            util.hard_update(self.target_actor, self.actor)
            util.hard_update(self.target_critic, self.critic)
            # pop-art
            self.target_y_mean = self.y_mean
            self.target_y_square_mean = self.y_square_mean
        else:
            util.soft_update(self.target_actor, self.actor, self.tau)
            util.soft_update(self.target_critic, self.critic, self.tau)
            # no sure how to update pop-art w.r.t. soft update
            self.target_y_mean = self.target_y_mean * (1.0 - self.tau) + self.y_mean * self.tau
            self.target_y_square_mean = self.target_y_square_mean * (1.0 - self.tau) + self.y_square_mean * self.tau

    def clip_grad(self, network):
        if self.grad_clip_mode is not None:
            if 'norm' in self.grad_clip_mode:
                torch.nn.utils.clip_grad_norm(network.parameters(), self.grad_clip_norm)
            else:
                for param in network.parameters():
                    param.grad.data.clamp_(-self.grad_clip_norm, self.grad_clip_norm)

    def update_actor_critic(self):
        # sample minibatch

        # per
        minibatch, w, e_id = self.exp.sample(self.total_step)
        if self.actor_update_mode=='dynamic':
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
        bat_agent = minibatch.agent if self.actor_update_mode=='dynamic' else None
        
        # update critic
        bat_a_o_ = Variable(bat_a.data.clone())
        for i in xrange(self.N):
            tmp_bat_o_ = bat_o_[:,i*self.n_s:(i+1)*self.n_s]
            tmp_bat_a_o_ = self.target_actor(tmp_bat_o_)
            bat_a_o_[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o_
        #bat_a_o_ = self.target_actor(bat_o_)

        Gt = bat_r

        if self.POPART:
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
        else:
            Gt[bat_not_done_mask] += self.GAMMA * self.target_critic(bat_o_, bat_a_o_)[bat_not_done_mask]

        Gt.detach_()
        # pop-art
        eval_o = y_delta * self.critic(bat_o, bat_a) + self.y_mean if self.POPART else self.critic(bat_o, bat_a)

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
        self.clip_grad(self.critic)

        self.optim_critic.step()

        
        # update actor
        self.critic.eval()


        # only update relevant agents
        #if self.DYNAMIC_ACTOR_UPDATE:
        if self.actor_update_mode=='dynamic':
            bat_a_o = Variable(bat_a.data.clone())
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
            # pop-art
            obj = torch.mean(y_delta * self.critic(bat_o, bat_a_o) + self.y_mean) if self.POPART else torch.mean(self.critic(bat_o, bat_a_o))
            self.optim_actor.zero_grad()
            obj.backward()
            self.clip_grad(self.actor)
            self.optim_actor.step()    
        # one by one
        elif 'obo' in self.actor_update_mode:
            bat_a_o_aux = Variable(bat_a.data.clone())
            for i in xrange(self.N):
                tmp_bat_o = bat_o[:,i*self.n_s:(i+1)*self.n_s]
                tmp_bat_a_o = self.target_actor(tmp_bat_o) if 'target' in self.actor_update_mode else self.actor(tmp_bat_o)
                bat_a_o_aux[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o
            for i in xrange(self.N):
                bat_a_o = Variable(bat_a_o_aux.data.clone())
                tmp_bat_o = bat_o[:,i*self.n_s:(i+1)*self.n_s]
                tmp_bat_a_o = self.actor(tmp_bat_o)
                bat_a_o[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o
                # pop-art
                obj = torch.mean(y_delta * self.critic(bat_o, bat_a_o) + self.y_mean) if self.POPART else torch.mean(self.critic(bat_o, bat_a_o))
                self.optim_actor.zero_grad()
                obj.backward()
                self.clip_grad(self.actor)
                self.optim_actor.step()    
        # default
        else:
            bat_a_o = Variable(bat_a.data.clone())
            for i in xrange(self.N):
                tmp_bat_o = bat_o[:,i*self.n_s:(i+1)*self.n_s]
                tmp_bat_a_o = self.actor(tmp_bat_o)
                bat_a_o[:,i*self.n_a:(i+1)*self.n_a] = tmp_bat_a_o
            # pop-art
            obj = torch.mean(y_delta * self.critic(bat_o, bat_a_o) + self.y_mean) if self.POPART else torch.mean(self.critic(bat_o, bat_a_o))
            self.optim_actor.zero_grad()
            obj.backward()
            self.clip_grad(self.actor)
            self.optim_actor.step()    

        
        # log
        if (self.epi+1) % self.model_log_freq == 0 and self.step==1:
            self.logger.scalar_summary('critic_loss', loss.data.cpu().numpy()[0], self.epi+1)
            for key, value in self.critic.named_parameters():
                key = key.replace('.', '/')
                self.logger.histo_summary('critic/'+key, value.data.cpu().numpy(), self.epi+1)
                self.logger.histo_summary('critic/'+key+'/grad', value.grad.data.cpu().numpy(), self.epi+1)
            for key, value in self.actor.named_parameters():
                key = key.replace('.', '/')
                self.logger.histo_summary('actor/'+key, value.data.cpu().numpy(), self.epi+1)
                self.logger.histo_summary('actor/'+key+'/grad', value.grad.data.cpu().numpy(), self.epi+1)

        self.critic.train()

    def test(self, dir=None, n=1):
        if dir is not None:
            self.env = wrappers.Monitor(self.orig_env, '{}/test_record'.format(dir), force=True, video_callable=lambda episode_id: True)

        title = {common.S_EPI:[], common.S_TOTAL_R:[]}
        df = pd.DataFrame(title)

        for self.epi in trange(n, desc='test epi', leave=True):
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
            s = pd.Series([self.epi, acc_r], index=[common.S_EPI, common.S_TOTAL_R])
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
        with open('{}/config{}.txt'.format(dir, suffix), 'w') as f:
            for item in self.config:
                f.write(str(item)+'\n')
        if self.POPART:
            with open('{}/popart{}.pkl'.format(dir, suffix), 'w') as f:
                pickle.dump([self.y_mean, self.y_square_mean], f)


    def load_actor(self, dir, suffix=''):
        self.actor.load_state_dict(torch.load('{}/actor{}.pt'.format(dir, suffix)))

    def load_critic(self, dir, suffix=''):
        self.critic.load_state_dict(torch.load('{}/critic{}.pt'.format(dir, suffix)))
        if self.POPART:
            with open('{}/popart{}.pkl'.format(dir, suffix)) as f:
                self.y_mean, self.y_square_mean = pickle.load(f)

    def get_data(self):
        return self.data

    

        


