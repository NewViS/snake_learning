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


class ConvNet(nn.Module):
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

BATCH_SIZE =  128
GAMMA = 0.99
OBSERVE = 50000
EXPLORE = 3000000
REPLAY_MEMORY = 1000000
EPS_START = 0.5
EPS_END = 0
TRAIN_LIFES = 150000
TAU = 0.05
LR =  1e-6

policy_net = ConvNet()
target_net = ConvNet()
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
eps_threshold=0.5

def select_action(state):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold -= (EPS_START-EPS_END)/(TRAIN_LIFES)
    if eps_threshold<EPS_END:
        eps_threshold=EPS_END
    steps_done += 1
    if sample < eps_threshold:
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
        self.model = ConvNet()
        load_model(self.model)
    

    def move(self, environment):
        BaseGameModel.move(self, environment)
        state = environment.state()
        state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            vixod = self.model(state).max(1)[1].view(1, 1)
        #vixod=self.model.select_action(state=state, epsilon=0, model=self.model.model)
        print(vixod)
        return self.action_all[vixod]

print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    model_game=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    env= model_game.prepare_training_environment()
    for i_episode in range(200000):
        # Initialize the environment and get it's state
        env.set_fruit()
        env.set_snake()
        observ = env.observation()
        state = env.state()
        action_all=Action.all()
        fake_reward_pa=0
        count_fl=1
        steps = 0
        circle_check = [-1] * 16
        circle_index = 0
        action_66=7
        dist=observ[8]
        score=0
        max_steps = 650
        reward=0
        time_out = 0
        hunger=200
        hang_out= 0
        
       
        
        while True :
            steps += 1
            time_out += 1
            state = torch.from_numpy(state).float().unsqueeze(0)
            print(state)
            action = select_action(state)
            old_action=action_66
        
            action_66=action_all[action]
            # if steps>1:
            #     if Action.is_reverse(old_action, action_66) :
            #         reward -=0.6
            # if action_66 != old_action:
            #     circle_check[circle_index % len(circle_check)] = action
            #     circle_index += 1
            #     for i in range(2, len(circle_check)):
            #         if ((circle_check[i-3] == 0 and
            #             circle_check[i-2] == 1 and
            #             circle_check[i-1] == 2 and
            #             circle_check[i] == 3) or

            #             (circle_check[i-3] == 3 and
            #             circle_check[i-2] == 2 and
            #             circle_check[i-1] == 1 and
            #             circle_check[i] == 0)):
            #                 #if fl==0:
            #                 #   reward -=7500
            #                 #  fl=1
            #             reward -= 0.6
            next_observ, fake_reward_fu, done,  = env.full_step(action_66)
            # if done and steps<15:
            #     reward -=1
            # elif  done:
            #     reward-=0.7
            # hunger-=1
            if done:
                reward= -1
            if fake_reward_fu==1 :
                time_out = 0
                if hang_out==0:
                    reward=1
                hunger=200        
                count_fl+=1
                if (count_fl> 10):
                    hang_out = math.ceil(0.4 * count_fl) + 2
                else:
                    hang_out = 6
            
            if(time_out >= math.ceil(count_fl * 0.7 + 10)):
                reward -= 0.5/count_fl
                time_out = 0
            if(hang_out == 0 and fake_reward_fu + fake_reward_pa == 0):
                
                if count_fl==1:
                    size = 2
                else:
                    size=count_fl
                reward += math.log(((size) + dist)/((size)+ next_observ[8])) / math.log(size)   
				
			
            # if hunger<0:
            #     reward -= 1     
            # if mindist>next_state[8]:
            #     reward += 0.5
            #     mindist = next_state[8]
            if(reward > 1):
                reward = 1
            elif (reward < -1):
                reward = -1
            score += reward
            reward = torch.tensor([reward],dtype=torch.float32, device="cpu")
            

            if done:
                next_observ = None
            else:
                dist=next_observ[8]
                next_observ = torch.tensor(next_observ, dtype=torch.float32, device="cpu").unsqueeze(0)

            # Store the transition in memory
            next_state=torch.from_numpy(env.state()).float().unsqueeze(0)
            memory.push(state, action,next_state, reward)

            # Move to the next state
            state = env.state()
            

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(steps + 1)
                
                break
            if(hang_out != 0):
                hang_out -= 1
        print(i_episode,count_fl,score,steps)

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