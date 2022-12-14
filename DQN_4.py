import torch
import os
from time import sleep
from tqdm import tqdm
#import gym
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy
#from torchvision import transforms
from random import random, randint, sample,choice
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from game import Game
from environment import Environment
from action import Action


class Memory:
    """
    Класс-буфер для сохранения результатов в формате
    (s, a, r, s', done).
    """
    def __init__(self, capacity):
        """
        :param capacity: размер буфера памяти.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """
        Данный метод сохраняет переданный элемент в циклический буфер.
        :param element: Элемент для сохранения.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.position] = element
            self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):
        """
        Данный метод возвращает случайную выборку из циклического буфера.
        :param batch_size: Размер выборки.
        :return: Выборка вида [(s1, s2, ... s-i), (a1, a2, ... a-i), (r1, r2, ... r-i),
         (s'1, s'2, ... s'-i), (done1,  done2, ..., done-i)],
            где i = batch_size - 1.
        """
        return list(zip(*sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)

class DeepQNetwork(nn.Module):
    """
    Класс полносвязной нейронной сети.
    """

    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions):

        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.relu = nn.ReLU()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = f.softmax(x, dim=-1)

        return x

class Agent:
    """
    Класс агента, обучающегося играть в игру.
    """
    def __init__(self,
                 env,
                 file_name,
                 max_epsilon=1,
                 min_epsilon=0.01,
                 target_update=1024,
                 memory_size=4096,
                 epochs=25,
                 batch_size=64):
        """
        :type env: gym.Env
        :param env gym.Env: Среда, в которой играет агент.
        :param file_name: Имя файла для сохранения и загрузки моделей.
        :param max_epsilon: Макимальная эпсилон для e-greedy police.
        :param min_epsilon: Минимальная эпсилон для e-greedy police.
        :param target_update: Частота копирования параметров из model в target_model.
        :param memory_size: Размер буфера памяти.
        :param epochs: Число эпох обучения.
        :param batch_size: Размер батча.
        """

        self.gamma = 0.97
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update

        self.file_name = file_name
        self.batch_size = batch_size

        self.device = torch.device("cpu")

        self.memory = Memory(capacity=memory_size)

        self.env = env

        self.model = DeepQNetwork(input_dims=19,
                                   fc1_dims=64,
                                   fc2_dims=64,
                                   n_actions=4).to(self.device)
        
        self.target_model = DeepQNetwork(input_dims=19,
                                          fc1_dims=64,
                                          fc2_dims=64,
                                          n_actions=4).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.epochs = epochs

        self.history = []

    def fit(self, batch):
        """
        Метод одной эпохи обучения. Скармливает модели данные,
        считает ошибку, берет градиент и делает шаг градиентного спуска.
        :param batch: Батч данных.
        :return: Возвращает ошибку для вывода в лог.
        """
        state, action, reward, next_state, done = batch

        # Распаковываем батч, оборачиваем данные в тензоры,
        # перемещаем их на GPU

        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).to(self.device)

        with torch.no_grad():
            # В этой части кода мы предсказываем максимальное
            # значение q-функции для следующего состояния,
            # см. ур-е Беллмана
            q_target = self.target_model(next_state).max(1)[0].view(-1)
            q_target = reward + self.gamma * q_target

            # Если следующее состояние конечное - то начисляем за него
            # награду за смерть, предусмотренную средой
            q_target[done] = -7500

        # Предсказываем q-функцию для действий из текущего состояния
        q = self.model(state).gather(1, action.unsqueeze(1))

        # Зануляем градиент, делаем backward, считаем ошибку,
        # делаем шаг оптимизатора
        self.optimizer.zero_grad()
        print(f"ЭТО ЕБУЧАЯ ОБЫЧНАЯ КУ:{q}\nЭТО ТАРГЕРНАЯ КУЖ:{q_target}\n")
        loss = self.criterion(q, q_target.unsqueeze(1))
        loss.backward()

        for param in self.model.parameters():
            param.data.clamp_(-1, 1)

        self.optimizer.step()
        
        return loss

    def train(self, max_steps=2**10, save_model_freq=100):
        """
        Метод обучения агента.
        :param max_steps: Из-за того, что в некоторых средах
            агент может существовать бесконечно долго,
            необходимо установить максимальное число шагов.
        :param save_model_freq: Частота сохранения параметров модели
        """

        max_steps = max_steps
        loss = 0
        action_all=Action.all()
        for epoch in tqdm(range(self.epochs)):
            step = 0
            done = False

            

            episode_rewards = []

            # Получаем начальное состояние среды
            self.env.set_fruit()
            self.env.set_snake()
            state = self.env.observation()
            count = 1
            score = 0
            reward=0
            fake_reward_pa=0
            count_fl=1
            circle_check = [-1] * 16
            circle_index = 0
            action_66=7
            mindist=state[8]
            fl=0
            # Играем одну игру до проигрыша, или пока не сделаем
            # максимальное число шагов
            while not done and step < max_steps:
                step += 1

                # Считаем epsilon для e-greedy police
                epsilon = (self.max_epsilon - self.min_epsilon) * (1 - epoch / self.epochs)

                # Выбираем действие с помощью e-greedy police
                action_choise = self.action_choice(state, epsilon, self.model)
                old_action=action_66
                action_66=action_all[action_choise]
                if count>1:
                    if Action.is_reverse(old_action, action_66) :
                        reward -=650
                if action_66 != old_action:
                    circle_check[circle_index % len(circle_check)] = action_choise
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
                        reward -= 600

                # Получаем новое состояние среды
                next_state, fake_reward_fu, done = self.env.full_step(action_66)
                if done and count<15:
                    reward -=10000
                elif  done:
                    reward-=7500
                if fake_reward_fu> fake_reward_pa:
                    reward+= 10000*count_fl
                    
                    count_fl+=1
                if mindist>next_state[8]:
                    score += 500
                    mindist = next_state[8]
                # if state[8]<next_state[8]:
                #     reward+= 500

                # if state[8]>next_state[8]:
                #     reward-= 1000
                # new_distance = abs(next_state_vector[0] - next_state_vector[2]) + \
                #     abs(next_state_vector[1] - next_state_vector[3])
                #
                # old_distance = abs(state_vector[0] - state_vector[2]) + \
                #     abs(state_vector[1] - state_vector[3])
                #
                # reward = reward - 5 * (new_distance - old_distance)

                episode_rewards.append(reward)

                if done or step == max_steps:
                    # Если игра закончилась, добавляем опыт в память

                    total_reward = sum(episode_rewards)
                    self.memory.push((torch.Tensor(state), action_choise, reward, torch.Tensor(next_state), done))

                    tqdm.write(f'Episode: {epoch},\n' +
                               f'Total reward: {total_reward},\n' +
                               f'Training loss: {loss:.4f},\n' +
                               f'Explore P: {epsilon:.4f},\n' +
                               f'Action: {action_choise}\n' +
                               f'Fruit: {count_fl}\n')

                else:
                    # Иначе - добавляем опыт в память и переходим в новое состояние
                    self.memory.push((torch.Tensor(state), action_choise, reward, torch.Tensor(next_state), done))
                    state = next_state
                    fake_reward_pa=fake_reward_fu
                    count+=1

            if epoch % self.target_update == 0:
                # Каждые target_update эпох копируем параметры модели в target_model,
                # согласно алгоритму
                self.target_model.load_state_dict(self.model.state_dict())

            if epoch % save_model_freq == 0:
                # Каждые save_model_freq эпох сохраняем модель
                # и играем тестовую игру, чтобы оценить модель
                
                self.save_model()

            if epoch > self.batch_size:
                # Поскольку изначально наш буфер пуст, нам нужно наполнить его,
                # прежде чем учить модель. Если буфер достаточно полон, то учим модель.
                loss = self.fit(self.memory.sample(batch_size=self.batch_size))

        self.save_model()
    def action_choice(self, state, epsilon, model):
            
        if random() < epsilon:
            # Выбираем случайное действие из возможных,
                # если случайное число меньше epsilon
            action = choice(torch.arange(4))
        else:
                # Иначе предсказываем полезность каждого действия из даного состояния
            action = model(torch.tensor(torch.Tensor(state).unsqueeze(0)).to(self.device)).view(-1)
                # И берем argmax() от предсказания, чтобы определить, какое действие
                # лучше всего совершить
            
            action = action.max(0)[1].item()

        return action
    def save_model(self, file_name=None):
        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'

        torch.save(self.model.state_dict(), path)

    def load_model(self, file_name=None):
            

        if file_name is None:
            file_name = self.file_name

        path = 'models/' + file_name + '.pth'

        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

class DQN_play(BaseGameModel):
    state_size=19
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self,):
        BaseGameModel.__init__(self, "dqn", "dqn", "dqn")
        self.model =  Agent(
                                    env=self.prepare_training_environment(),
                                    file_name='snake_7.2',
                                    max_epsilon=1,
                                    min_epsilon=0,
                                    target_update=2000,
                                    epochs=2**15,
                                    batch_size=16,
                                    memory_size=8000)
        self.model.load_model()
    

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        state = torch.tensor((environment.observation())).float()
        vixod=self.model.action_choice(state=state, epsilon=0, model=self.model.model)
        print(vixod)
        return self.action_all[vixod]

print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    model=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    agent = Agent(
    env=model.prepare_training_environment(),
    file_name='snake_7.2',
    max_epsilon=1,
    min_epsilon=0,
    target_update=2000,
    epochs=2**15,
    batch_size=16,
    memory_size=8000)
    agent.train()
if a==1:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
    