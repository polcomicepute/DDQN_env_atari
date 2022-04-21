
from ast import arg
from audioop import avg
# from agent import FloatTensor
# from memory_hm import ReplayBuffer
from utils.atari_wrapper import make_atari, wrap_deepmind
from itertools import count
import torch
import sys
import numpy as np
import random
from model import DDQN
from utils.memory import Transition, ReplayBuffer
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
np.set_printoptions(threshold=sys.maxsize)
from gym import wrappers
import os

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('./log_dir')

class Agent(object):
    def __init__(self,  cfg):    #, cfg):
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = 'gym-results-hm/'
        self.env_id = cfg['env_id']
        self.env = make_atari(env_id= self.env_id )
        self.env = wrappers.Monitor(self.env, directory=self.save_dir, force=True, video_callable=False)
        self.env = wrap_deepmind(self.env)
        self.n_actions = self.env.action_space.n
        self.replay_start = cfg['replay_start']
        self.log_freq = cfg['log_freq']
        self.save_freq = cfg['save_freq']
        # self.eval_freq = cfg['']
        self.target_update = cfg['target_update']

        # * Epsilon
        self.start_eps = cfg['epsilon']['start']
        self.end_eps = cfg['epsilon']['end']
        self.decay_step = cfg['epsilon']['decay_step']
        img_h, img_w, img_c = self.env.observation_space.shape
        self.state_size = [cfg['hist_len']*img_c, img_h, img_w]

        # * Model
        self.behaviour_Qnet = DDQN(input_size=self.state_size[0], output_size=self.n_actions).to(self.device)
        self.target_Qnet = DDQN(input_size=self.state_size[0], output_size=self.n_actions).to(self.device)
        self.target_Qnet.load_state_dict(self.behaviour_Qnet.state_dict())
        self.target_Qnet.eval()
        self.optimizer = torch.optim.Adam(self.behaviour_Qnet.parameters(), lr = float(cfg['optim_params']['lr']))

        self.gamma = cfg['gamma']

        # * Replay Buffer
        self.replay_buffer = ReplayBuffer(cfg['replay_memory'])
        self.batch_size = cfg['batch_size']

    def epsilon_decay(self, curr_step):
        temp = (self.start_eps - self.end_eps)/self.decay_step
        if curr_step > self.decay_step:
            eps = self.end_eps
        else:
            eps = self.start_eps - temp*curr_step
        return eps
    
            # with torch.no_grad():
            # action = self.env.action_space.sample()
    
    
    def get_action(self, eps, state):
        if random.random() >= eps: #get action from argmax_Q
            state = state.to(self.device, dtype= torch.float)
            action = self.behaviour_Qnet(state.unsqueeze(0)).max(1)[1] #.detach()
        else:
            action = torch.tensor([random.randrange(self.n_actions)], device=self.device, dtype=torch.long)
        return action


    def train(self):
        mean_episode_reward = -float('nan')
        best_mean_episode_reward = -float('inf')
        # * obs reset 
        epi_reward = 0
        all_rewards =[]
        avg_rewards=[]
        epi_num = 0
        state = np.zeros(self.state_size )
        obs = self.env.reset()
        loss =None

        state = torch.tensor(np.vstack([state[1:], (obs.transpose(2, 0, 1))/255]) )#, dtype = torch.float)

        for t in count():
            self.behaviour_Qnet.train()
            # self.target_Qnet.train()

            eps = self.epsilon_decay(t)
            
            action = self.get_action(eps, state)
            next_obs, reward, done, _ = self.env.step(action)
            self.env.render()

            next_state = torch.vstack([state[1:], (torch.from_numpy(next_obs.transpose(2, 0, 1)))/255])
            # if np.array_equal(state, next_state):
            #     print("same") #! NOT SAME
            epi_reward +=reward
            reward = torch.tensor(reward, device = self.device)

            self.replay_buffer.push(state, action, reward, next_state)

            if done:
                state = np.zeros(self.state_size )
                obs = self.env.reset()
                state = torch.tensor(np.vstack([state[1:], (obs.transpose(2, 0, 1))/255]) )
                all_rewards.append(epi_reward)
                epi_reward =0
                epi_num += 1

                avg_rewards =float(np.mean(all_rewards[-100:]))
                writer.add_scalar('mean_episode_reward',avg_rewards, epi_num)

            else:
                state = next_state  


            # * Sampling Batch -> train
            if len(self.replay_buffer)> self.batch_size and t > self.replay_start:
                transitions = self.replay_buffer.sample(self.batch_size)
                batch =  Transition(*zip(*transitions))

                state_batch = torch.stack(batch.state).to(self.device, dtype= torch.float)
                action_batch = torch.stack(batch.action).to(self.device)
                reward_batch = torch.stack(batch.reward).to(self.device, dtype= torch.float)

                # * Prediction 
                pred  = self.behaviour_Qnet(state_batch).gather(1, action_batch).squeeze(1)

                # * Target
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
                non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None]).to(self.device, dtype= torch.float)
                # print(non_final_mask.shape, non_final_next_states.shape)
                target_Q = torch.zeros(self.batch_size, device=self.device)#.unsqueeze(1)
                argmax_a = self.behaviour_Qnet(non_final_next_states).max(1, keepdim=True)[1]

                target_Q[non_final_mask] = self.target_Qnet(non_final_next_states).gather(1, argmax_a).squeeze(1)
                target = reward_batch+ self.gamma* target_Q

                loss = (pred - target.detach()).pow(2).mean()

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
            
                writer.add_scalar('Loss/train', loss.item(), t)

                if t % self.target_update ==0:
                    self.target_Qnet.load_state_dict(self.behaviour_Qnet.state_dict())


            if t % self.log_freq ==0:
                if len(all_rewards)>0:
                    writer.add_scalar('reward per epi', all_rewards[-1], epi_num)
                    if loss!=None:
                        print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (t, np.mean(all_rewards[-10:]), loss, epi_num))
                if len(all_rewards) > 40:
                    if max(best_mean_episode_reward, avg_rewards) > best_mean_episode_reward:
                        best_mean_episode_reward = max(best_mean_episode_reward, avg_rewards)
                        
                        writer.add_scalar('best_mean_episode_reward/train', best_mean_episode_reward, t)
                        self.save('Best_QNetwork.pth.tar')
            if t % self.save_freq == 0:
                    self.save('QNetwork.pth.tar')

    def save(self, name):
        modelpath =os.path.join(self.save_dir, name)
        torch.save(self.behaviour_Qnet.state_dict(), modelpath)
        print("Saved to model to {}".format(modelpath))

    def load(self, name):
        modelpath =os.path.join(self.save_dir, name)
        state_dict = torch.load(modelpath, map_location=torch.device('cpu'))
        self.behaviour_Qnet.load_state_dict(state_dict)
                
    def eval(self, n_episodes):
        self.behaviour_Qnet.to('cpu').eval()
        self.target_Qnet.to('cpu').eval()
        self.load('QNetwork.pth.tar')
        epi_reward = 0
        all_rewards =[]


        for i_ep in range(n_episodes):
            state = np.zeros(self.state_size)
            obs = self.env.reset()
            state = torch.tensor(np.vstack([state[1:], (obs.transpose(2, 0, 1))/255])).to('cpu')#, dtype = torch.float)

            for t in count():
                action = self.get_action(0, state)
                next_obs, reward, done, _ = self.env.step(action)
                self.env.render()
                next_state = torch.vstack([state[1:], (torch.from_numpy(next_obs.transpose(2, 0, 1)))/255]).to('cpu')

                epi_reward += reward

                if done:
                    all_rewards.append(epi_reward)
                    print('[Test] episode: %3d, episode_reward: %5f' % (i_ep, epi_reward))
                    epi_reward = 0
                    break
                else:
                    state = next_state  
            

        print("avg reward: %5f" % (np.mean(all_rewards)))
        self.env.reset()
        


if __name__ == '__main__':
    agt = Agent('PongNoFrameskip-v4')
    agt.train()


        
