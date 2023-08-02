import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from itertools import count
from torch.distributions import Categorical

capacity=1000000

batch_size=64
update_iteration=200
tau=0.001
# tau for soft updating

gamma=0.99
directory = './'
hidden1=20
hidden2=64

import numpy as np
import random
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Replay_buffer():

    def __init__(self, max_size=capacity):
        """
        This code snippet creates a Replay buffer. 
        The buffer is designed to store a maximum number of transitions, indicated by the "size" parameter. 
        When the buffer reaches its maximum capacity, it starts dropping the oldest memories to make space for new ones.
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:    #If storage is full
            self.storage[int(self.ptr)] = data     #Store data where the pointer currently points at
            self.ptr = (self.ptr + 1) % self.max_size    #Increment pointer. if pointer is at last index, the modulus ensures it reutrn to begiining
        else:
            self.storage.append(data)   #If buffer has not yet been filled, keep appending to buffer

    def sample(self, batch_size):

        ind = np.random.randint(0, len(self.storage), size=batch_size) #Choose random batch of indices
        state, next_state, probs, action, reward, done = [], [], [], [], [], []     #Initialise list for sampled experience

        for i in ind:
            st, n_st, prbs, act, rew, dn = self.storage[i]       #iterate through indices and retrieve sampled experience
            state.append(np.array(st, copy=False))
            next_state.append(np.array(n_st, copy=False))
            probs.append(np.array(prbs, copy=False))
            action.append(np.array(act, copy=False))
            reward.append(np.array(rew, copy=False))
            done.append(np.array(dn, copy=False))

        #return batch of sampled experiences

        return np.array(state), np.array(next_state), np.array(probs), np.array(action), np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self, n_states, action_dim, hidden1):
        super(Actor, self).__init__()
        self.net = nn.Sequential(                       #Initialise our actor netowrk with 2 hidden layers and ReLU activation
            nn.Linear(n_states, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, hidden1), 
            nn.ReLU(), 
            nn.Linear(hidden1, action_dim)
        )
        
    def forward(self, state):
        
        action = self.net(state)                                       #Get action probabilities from actor network
        softmax_tensor = torch.softmax(action, dim = 0)                #Normalise them such that they sum to 1
        cat_dist = Categorical(probs=softmax_tensor)                   #Use our probabilities to create a categorical distribution
        sampled_action = cat_dist.sample()                             #Sample from the categorical distribution



        return action,sampled_action         #Return action probabilities, as well as the sampled action

class Critic(nn.Module):
   
    def __init__(self, n_states, action_dim, hidden2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states + action_dim, hidden2),  #Initialise our actor netowrk with 2 hidden layers and ReLU activation
            nn.ReLU(), 
            nn.Linear(hidden2, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, hidden2), 
            nn.ReLU(), 
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))  #Return our Q value output


class DDPG(object):
    def __init__(self, state_dim, action_dim):

        
        self.replay_buffer = Replay_buffer()     #initialise replay buffer
        
        self.actor = Actor(state_dim, action_dim, hidden1).to(device)           #initialise actor network
        self.actor_target = Actor(state_dim, action_dim,  hidden1).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-3)

        self.critic = Critic(state_dim, 1,  hidden2).to(device)                 #initialise critic network
        self.critic_target = Critic(state_dim, 1,  hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=2e-2)
        

        #initialise hyperparameters

        self.num_critic_update_iteration = 0          
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
  
        state = torch.FloatTensor(state.reshape(1, -1)).to(device) #Pass the state vector to the actor network
        
        return self.actor(state)[0].cpu().data.numpy().flatten(), self.actor(state)[1].cpu().data.numpy().flatten()     #Return the action and probability vector 


    def update(self):


        for it in range(update_iteration):
            # For each Sample in replay buffer batch
            state, next_state, probs, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device) #convert numpy array in replay buffer to tensor
            probs = torch.FloatTensor(probs).to(device)
            action = torch.FloatTensor(action).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(1-done).to(device)
            reward = torch.FloatTensor(reward).to(device)

            # Compute the target Q value
            

            target_Q = self.critic_target(next_state, np.reshape(self.actor_target(next_state)[1],(-1,1)))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            
            #Initialise out categorical distribution using the probabilities we obtained from the actor network.
            dist = Categorical(logits=probs)

            

            # Compute actor loss as the negative mean Q value using the critic network and the actor network
            actor_loss = -dist.log_prob(action).mean()*self.critic(state, np.reshape(self.actor_target(next_state)[1],(-1,1))).mean()
            

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            
            """
            Update the frozen target models using 
            soft updates, where 
            tau,a small fraction of the actor and critic network weights are transferred to their target counterparts. 
            """
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
           
            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1