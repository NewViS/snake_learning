import torch
import torch.nn as nn
import torch.nn.functional as F
import random 
import torch.optim as optim
from collections import namedtuple, deque
from base_game_model import BaseGameModel
from action import Action
from pygame.locals import *
import random
from constants import Constants
from game import Game
import copy
from environment import Environment
import math
from action import Action
import numpy as np

BUFFER_SIZE = 8000 
BATCH_SIZE = 16         
GAMMA = 0.97           
TAU = 1e-3              
LR = 1e-3               
def possible_action(environment):
    return [Action.left_neighbor(environment.snake_action), environment.snake_action,
                   Action.right_neighbor(environment.snake_action)]
class DQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 11)
        self.fc2 = nn.Linear(11, 11)
        self.fc3 = nn.Linear(11, action_size)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        
        self.qnetwork_local = DQN(state_size, action_size, seed).to("cpu")
        self.qnetwork_target = DQN(state_size, action_size, seed).to("cpu")
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        
        self.qnetwork_local.eval()
        state=torch.Tensor(state).to("cpu")
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        
        if random.random() < eps:
            return random.choice(torch.arange(self.action_size))
                
        else:
            return np.argmax(action_values.cpu().data.numpy())
        

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1,actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience((state), (action), (reward), (next_state), (done))
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to("cpu")
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to("cpu")
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to("cpu")
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to("cpu")
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to("cpu")
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)




class DQN_trainer(BaseGameModel):

    save_path='checkpoint.pth'
    episodes=2**15
    observation_space=19
    action_space=4
    max_t=1500
    eps_start=1
    eps_end=0
    eps_decay=0.995
    action=Action.all()
    def __init__(self):
        BaseGameModel.__init__(self, "dqn_trainer", "dqn_trainer", "dqn_trainer")
    def move(self, env,):
        BaseGameModel.move(self, env)
        agent = Agent(self.observation_space, self.action_space, 0)
        self.train_dqn(agent, env, self.episodes, self.max_t, self.eps_start, self.eps_end, self.save_path )
    def train_dqn(self,agent, env, episodes, max_t, eps_start, eps_end, save_path):
        scores = []
        eps = eps_start
        done=9

        
        
        for j in range(1, episodes + 1):
        
            env.set_fruit()
            env.set_snake()
            state = env.observation()
            count = 1
            score = 0
            reward=0
            fake_reward_pa=1
            count_fl=1
            circle_check = [-1] * 16
            circle_index = 0
            action_66=7
            for t in range(max_t):
                action_choise = agent.act(state, eps)
                old_action=action_66
                action_66=self.action[(action_choise.item())%4]
                if count>1:
                    if Action.is_reverse(old_action, action_66) :
                        reward -=650
                if action_66 != old_action:
                    circle_check[circle_index % len(circle_check)] = (action_choise.item())%4
                    circle_index += 1
                for i in range(2, len(circle_check)):
                    if ((circle_check[i-3] == 0 and
                        circle_check[i-2] == 1 and
                        circle_check[i-1] == 2 and
                        circle_check[i] == 3) or

                        (circle_check[i-3] == 3 and
                        circle_check[i-2] == 2 and
                        circle_check[i-1] == 1 and
                        circle_check[i] == 0)):
                        reward -=10000
                next_state, fake_reward_fu, done = env.full_step(action_66)

                if done and count<15:
                    reward -=10000
                elif  done:
                    reward-=7500
                if fake_reward_fu> fake_reward_pa:
                    reward+= 3500*count_fl
                    
                    count_fl+=1
                if state[8]<next_state[8]:
                    reward+= 1000
                agent.step(state, (action_choise.item())%4, reward, next_state, done)
                state = next_state
                fake_reward_pa=fake_reward_fu
                count+=1
                if done:
                    break          
            scores.append(score)             
            eps = max(eps_end, (1 - j / (episodes + 1))*eps)  
        torch.save(agent.qnetwork_local.state_dict(), save_path)


class DQN_play(BaseGameModel):
    state_size=19
    action_size=4
    pth_path= 'checkpoint.pth'
    action=Action.all()
    def __init__(self,):
        BaseGameModel.__init__(self, "dqn", "dqn", "dqn")
        self.model = DQN(self.state_size, self.action_size, 0)
        self.model.load_state_dict(torch.load(self.pth_path))
    

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        state = torch.tensor((environment.observation())).float()
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        return self.action[np.argmax(action_values.data.numpy())]




print(f"?????????????? ?????????? 0-???????? ????????????????, 1-???????? ????????")
a = int(input())
if a==0:
    model=DQN_trainer()
    model.move(model.prepare_training_environment())
if a==1:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
    
    