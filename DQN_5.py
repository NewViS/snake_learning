import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from game import Game
from environment import Environment
from action import Action

EPISODES = 2**15  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.01  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.80  # Q-learning discount factor
LR = 0.0005  # NN optimizer learning rate
HIDDEN_LAYER = 24  # NN hidden layer size
BATCH_SIZE = 128  # Q-learning batch size


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(19, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 16)
        self.l3 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x



model = Network()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return torch.LongTensor([[random.randrange(4)]])

def run_episode(e, environment):
    environment.set_fruit()
    environment.set_snake()
    state = environment.observation()
    action_all=Action.all()
    fake_reward_pa=0
    count_fl=1
    steps = 0
    circle_check = [-1] * 16
    circle_index = 0
    action_66=7
    mindist=state[8]
    max_steps = 650
    reward=0
    while True and steps < max_steps:
        steps += 1
        
        action = select_action(torch.FloatTensor([state]))
        old_action=action_66
       
        action_66=action_all[action]
        if steps>1:
            if Action.is_reverse(old_action, action_66) :
                reward -=6
        if action_66 != old_action:
            circle_check[circle_index % len(circle_check)] = action
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
                        #if fl==0:
                         #   reward -=7500
                          #  fl=1
                    reward -= 6
        
        next_state, fake_reward_fu, done = environment.full_step(action_66)
        if done and steps<15:
            reward -=10
        elif  done:
            reward-=7.5
        if fake_reward_fu> fake_reward_pa:
            reward+= 10*count_fl
                    
            count_fl+=1
            print(count_fl)
        if mindist>next_state[8]:
            reward += 5
            mindist = next_state[8]
        # zero reward when attempt ends
        
        
        memory.push((torch.FloatTensor([state]),
                     action,  # action is already a tensor
                     torch.FloatTensor([next_state]),
                     torch.FloatTensor([reward])))

        learn()
        if done:
            break
        state = next_state

        


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)
    print(loss)
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    model_game=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    for e in range(EPISODES):
        run_episode(e, model_game.prepare_training_environment())

    print('Complete')
    plt.ioff()
    plt.show()