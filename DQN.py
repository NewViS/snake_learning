from base_game_model import BaseGameModel
from action import Action
from pygame.locals import *
import random
from constants import Constants
from game import Game
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.autograd import Variable
import numpy as np
import random
import copy
from environment import Environment
import math




class DQNModel(nn.Module):
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        super(DQNModel, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(n_state, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden, n_action)
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.model_target= copy.deepcopy(self.model)
    
    def target_predict(self,s):
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))
    
    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, s):
            return self.model(torch.Tensor(s))
    
    def replay(self,memory, replay_size, gamma):
        if len(memory)>=replay_size:
            replay_data=random.sample(memory,replay_size)
            states=[]
            td_targets=[]
            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values =self.predict(state).tolist()
                if is_done:
                    q_values[action]= reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action]= reward+ gamma*torch.max(q_values_next).item()
                td_targets.append(q_values)
            loss =self.update(states,td_targets)
            return loss





# def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
#     def policy_function(state):
#         if random.random() < epsilon:
#             return random.randint(0, n_action - 1)
#         else:
#             q_values = estimator.predict(state)
#             return torch.argmax(q_values).item()
#     return policy_function

# def q_learning(env, estimator, n_episode,replay_size,target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    
#     for episode in range(n_episode):
#         if episode % target_update==0:
#             estimator.copy_target()
#         policy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
#         state = env.observation()
#         is_done = False
#         modified_reward=0
#         count=0
#         while not is_done:
#             # action = policy(state)
#             env.action=policy(state)
#             next_state, reward, is_done = env.full_step(env.action)
#             total_reward_episode[episode] += reward
#             count+=1
#             modified_reward -= 0.25

#             if is_done:
#                 modified_reward -= 10
#                 if count<=15:
#                      modified_reward -= 90
            
#             modified_reward+=math.sqrt(state[-1])*3.5
#             memory.append((state,env.action,next_state,modified_reward,is_done))

#             if is_done:
                
#                 break

#             estimator.replay(memory, replay_size,gamma)
#             state = next_state


#         print('Episode: {}, total reward: {}, epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))

#         epsilon = max(epsilon * epsilon_decay, 0.01)
# batch_size=32

# n_state = 4
# n_action = 3
# n_hidden = 50
# lr = 0.01
# gamma=0.99
# init_epsilon = 0.1
# final_epsilon = 1e-4
# saved_path = 'trained_models'
# torch.manual_seed(123)
# estimator = DQNModel(n_state,n_action,n_hidden,lr)
# memory=deque(maxlen=10000)
# n_episode = 600
# replay_size=20
# target_update=10
# total_reward_episode = [0] * n_episode
# env = Environment(width=300,height=330)
# q_learning(env, estimator, n_episode,replay_size, gamma=.9, epsilon=.3)

# class DQN(BaseGameModel):
#     saved_path = 'trained_models'
#     model = torch.load("{}/final".format(saved_path))
#     env = Environment(width=300,height=330)
#     def __init__(self):
#         BaseGameModel.__init__(self, "DQN", "dqn", "dqn")

#     def move(self, environment):
        
        
        
#         BaseGameModel.move(self, environment)
#         prediction = self.model(environment.state)
#         action = torch.argmax(prediction).item()
#         #predicted_action = self._predict(environment, self.model)
#         return action



class DQN_trainer(BaseGameModel):

    n_state = 5
    n_action = 4
    n_hidden = 4
    
    lr = 5e-4 
    gamma = 0.99
    init_epsilon = 0.5
    final_epsilon = 1e-4
    n_episode = 20000
    memory_size = int(1e5)
    saved_path = 'trained_models'
    torch.manual_seed(123)
    estimator = DQNModel(n_state,n_action,n_hidden,lr)
    memory=deque(maxlen=memory_size)
    
    replay_size=50
    target_update=10
    total_reward_episode = [0] * n_episode
    all_action=Action.all()
    action = None

    def __init__(self):
        BaseGameModel.__init__(self, "dqn_trainer", "dqn_trainer", "dqn_trainer")

    def move(self, environment):
        BaseGameModel.move(self, environment)
        
        self.q_learning(environment, self.estimator, self.n_episode,self.replay_size, gamma=0.7)

    def gen_epsilon_greedy_policy(self,estimator, epsilon, n_action):
        def policy_function(state):
            if random.random() < epsilon:
                #print("H")
                return random.randint(0, n_action - 1)
                print("H")
            else:
                #print("Q")
                q_values = estimator.predict(state)
                #print(q_values)
                return torch.argmax(q_values).item()
        return policy_function
    
    def q_learning(self,env, estimator, n_episode,replay_size,target_update=10, gamma=0.7):
    
        for episode in range(n_episode):
            if episode % target_update==0:
                estimator.copy_target()
            epsilon = self.final_epsilon + (n_episode - episode) * (self.init_epsilon - self.final_epsilon) / self.n_episode
            policy = self.gen_epsilon_greedy_policy(estimator, epsilon, self.n_action)
            if self.action is None:
                self.action = random.choice(Action.all())
            state = env.observation(self.action)
            is_done = False
            modified_reward=0
            count=0
            loss=0
            fruit_counter=0
            env.snake_lenght=0
            while not is_done:
                
                num_action = policy(state)
                self.action=self.all_action[num_action]
                next_state, reward, is_done = env.full_step(self.action)
                print(reward,"\t",is_done)
                self.total_reward_episode[episode] += reward
                count+=1
                modified_reward -= 100
                
                
                
                #print(modified_reward,"\t",episode,"\t",epsilon,"\t","\t")
                self.memory.append([state,num_action,next_state,modified_reward,is_done])

                if is_done:
                    
                    break

                loss=estimator.replay(self.memory, replay_size,gamma)
                state = next_state
            if (episode+1) % 10000 == 0:
                print("hi")
                torch.save(estimator.model, "{}/{}".format(self.saved_path, episode+1))

            #print('Episode: {}, total reward: {}, epsilon: {}'.format(episode, modified_reward, epsilon))

            #epsilon = max(epsilon * epsilon_decay, 0.01)
        torch.save(estimator.model, "{}/final".format(self.saved_path))

class DQN(BaseGameModel):
    saved_path = 'trained_models'
    model = torch.load("{}/final".format(saved_path))
    model.eval()
    env = Environment(width=300,height=330)
    all_action=Action.all()
    def __init__(self):
        BaseGameModel.__init__(self, "dqn", "dqn", "dqn")

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        prediction = self.model(torch.Tensor(environment.observation()))
        action = torch.argmax(prediction).item()
        
        #predicted_action = self._predict(environment, self.model)
        return self.all_action[action]



print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    model=DQN_trainer()
    model.move(model.prepare_training_environment())
if a==1:
    while True:
                Game(game_model=DQN(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
    
    