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
from collections import namedtuple, deque
from itertools import count
def save_model(model, file_name=None):
    if file_name is None:
        file_name = 'snake_7.2'

    path = 'models/' + file_name + '.pth'

    torch.save(model.state_dict(), path)

def load_model(model, file_name=None):
    if file_name is None:
        file_name = 'snake_7.2'

    path = 'models/' + file_name + '.pth'

    model.load_state_dict(torch.load(path))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, n_actions)
        self.rl=nn.ReLU(inplace=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x=self.layer1(x)
        x=self.rl(x)
        x=self.layer2(x)
        x=self.rl(x)
        x=self.layer3(x)
        return x

BATCH_SIZE =  128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

policy_net = DQN(19, 4).to("cpu")
target_net = DQN(19, 4).to("cpu")
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device="cpu", dtype=torch.long)


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device="cpu", dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device="cpu")
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

class DQN_play(BaseGameModel):
    state_size=19
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self,):
        BaseGameModel.__init__(self, "dqn", "dqn", "dqn")
        self.model = DQN(19, 4).to("cpu")
        load_model(self.model)
    

    def move(self, environment):
        BaseGameModel.move(self, environment)
        state = environment.observation()
        state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            vixod = self.model(state).max(1)[1].view(1, 1)
        #vixod=self.model.select_action(state=state, epsilon=0, model=self.model.model)
        print(vixod)
        return self.action_all[vixod]

print(f"?????????????? ?????????? 0-???????? ????????????????, 1-???????? ????????")
a = int(input())
if a==0:
    model_game=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    env= model_game.prepare_training_environment()
    for i_episode in range(10000):
        # Initialize the environment and get it's state
        env.set_fruit()
        env.set_snake()
        state = env.observation()
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
        hunger=200
        state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        while True and steps < max_steps:
            steps += 1
            action = select_action(state)
            old_action=action_66
        
            action_66=action_all[action]
            if steps>1:
                if Action.is_reverse(old_action, action_66) :
                    reward -=0.6
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
                        reward -= 0.6
            next_state, fake_reward_fu, done = env.full_step(action_66)
            if done and steps<15:
                reward -=1
            elif  done:
                reward-=0.7
            hunger-=1
            if fake_reward_fu> fake_reward_pa:
                reward+= 2
                hunger=200        
                count_fl+=1
            if hunger<0:
                reward -= 1     
            if mindist>next_state[8]:
                reward += 0.5
                mindist = next_state[8]
            reward = torch.tensor([reward],dtype=torch.float32, device="cpu")
            

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device="cpu").unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # ????? ??? ?? ?? + (1 ????? )?????
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(steps + 1)
                
                break
        print(i_episode,count_fl,reward,steps)

        if i_episode % 1000==0:
            save_model(policy_net)
    save_model(policy_net)


else:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
