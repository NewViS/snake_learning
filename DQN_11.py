import torch
import os
from time import sleep
from tqdm import tqdm
import gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from copy import deepcopy
from torchvision import transforms
from random import random, randint, sample,choice
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from game import Game
from environment import Environment
from action import Action
from collections import deque


def conv_block(in_channels, out_channels, depth=2, pool = False, drop=False, prob=0.2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    layers.append(nn.ReLU(inplace=True))
    for i in range(depth-1):
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    if pool:
        layers.append(nn.MaxPool2d(2))
    if drop:
        layers.append(nn.Dropout2d(p=prob))
    return nn.Sequential(*layers)
class DeepQNetwork(nn.Module):
    """
    Класс полносвязной нейронной сети.
    """
    def __init__(self,):
        super().__init__()
        self.conv1 = conv_block(in_channels=4,out_channels=32, pool=True)
        self.conv2 = conv_block(in_channels=32,out_channels=64, pool=True)
        self.fcn = nn.Sequential(nn.AvgPool2d(3), nn.Flatten(), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128,4))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.fcn(x)
        return(x)

class DQN():
    def __init__(self, n_action, lr=1e-6):
        self.criterion = nn.MSELoss()
        self.model = DeepQNetwork(n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, y_predict, y_target):
        """
        Update the weights of the DQN given a training sample
        @param y_predict:
        @param y_target:
        @return:
        """
        loss = self.criterion(y_predict, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, s):
        """
        Compute the Q values of the state for all actions using the learning model
        @param s: input state
        @return: Q values of the state for all actions
        """
        return self.model(torch.Tensor(s))

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay
        @param memory: a list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        @return: the loss
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*replay_data)

            state_batch = torch.cat(tuple(state for state in state_batch))
            next_state_batch = torch.cat(tuple(state for state in next_state_batch))
            q_values_batch = self.predict(state_batch)
            q_values_next_batch = self.predict(next_state_batch)

            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])

            action_batch = torch.from_numpy(
                np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))

            q_value = torch.sum(q_values_batch * action_batch, dim=1)

            td_targets = torch.cat(
                tuple(reward if terminal else reward + gamma * torch.max(prediction) for reward, terminal, prediction
                    in zip(reward_batch, done_batch, q_values_next_batch)))

            loss = self.update(q_value, td_targets)
            return loss

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action - 1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


class DQN_trainer(BaseGameModel):

    image_size = 84
    batch_size = 32
    lr = 1e-6
    gamma = 0.99
    init_epsilon = 0.1
    final_epsilon = 1e-4
    n_iter = 2000000
    memory_size = 50000
    saved_path = 'trained_models'
    n_action = 4
    all_action=Action.all()
    memory = deque(maxlen=memory_size)
    estimator = DQN(n_action)
    env = BaseGameModel.prepare_training_environment()
    def __init__(self):
        BaseGameModel.__init__(self, "dqn_trainer", "dqn_trainer", "dqn_trainer")
    def train(self, env,):
        for iter in range(self.n_iter):
            step = 0
            done = False            
            episode_rewards = []
            # Получаем начальное состояние среды
            self.env.set_fruit()
            self.env.set_snake()
            observ = self.env.observation()
            state=self.env.state()
            count = 1
            dist=observ[8]
            time_out = 0
            hang_out= 0
            action_66=7
            reward=0
            fake_reward_pa=0
            count_fl=1
            hunger =200
            while not done and hunger!=0:

                epsilon = self.final_epsilon + (self.n_iter - iter) * (self.init_epsilon - self.final_epsilon) / self.n_iter
                policy = gen_epsilon_greedy_policy(self.estimator, epsilon, self.n_action)
                action = policy(state)
                next_image, reward, is_done = env.next_step(action)
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
                self.memory.append([state, action, next_state, reward, is_done])
                loss = self.estimator.replay(self.memory, self.batch_size, self.gamma)
                state = next_state
                print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}".format(
                        iter + 1, self.n_iter, action, loss, epsilon, reward))
                if iter+1 % 10000 == 0:
                    torch.save(self.estimator.model, "{}/{}".format(self.saved_path, iter+1))

    torch.save(estimator.model, "{}/final".format(saved_path))


class DQN_play(BaseGameModel):
    # saved_path = 'trained_models'
    # model = torch.load("{}/final".format(saved_path))
    # model.eval()
    # env = Environment(width=300,height=330)
    # all_action=Action.all()
    state_size=19
    action_size=3
    pth_path= 'checkpoint.pth'
    action_all=Action.possible()
    def __init__(self,):
        BaseGameModel.__init__(self, "dqn", "dqn", "dqn")
        self.model = DQN(self.state_size, self.action_size, 0)
        self.model.load_state_dict(torch.load(self.pth_path))
    

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        action=  possible_action(environment)
        
        state = torch.tensor((environment.observation())).float()
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        print(action_values )
        #environment.full_step(action[np.argmax(action_values.data.numpy())])
        #print(action_values)
        #print(action_values.data.numpy())
        return action[np.argmax(action_values.data.numpy())]
    # def move(self, environment):
    #     BaseGameModel.move(self, environment) 
    #     prediction = self.model(torch.Tensor(environment.observation()))
    #     action = torch.argmax(prediction).item()
        
    #     #predicted_action = self._predict(environment, self.model)
    #     return self.all_action[action]



print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    model=DQN_trainer()
    
if a==1:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
    