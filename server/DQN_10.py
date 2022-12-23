import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import math
from random import random, sample, choice
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
#from game import Game
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



    
class Agent:
    """
    Класс агента, обучающегося играть в игру.
    """
    def __init__(self,
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


        self.model = DeepQNetwork().to(self.device)
        
        self.target_model = DeepQNetwork().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()

        self.epochs = epochs

        self.history = []

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

    def load_model(self, file_name=None):
        if file_name is None:
            file_name = self.file_name
        path = 'models/' + file_name + '.pth'
        print('ok')
        self.model.load_state_dict(torch.load(path))
        print('ok')
        self.target_model.load_state_dict(torch.load(path))
        print('ok')


class DQN_play(BaseGameModel):
    state_size=19
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self):
        self.model =  Agent(
                                    file_name='snake_7.2',
                                    max_epsilon=1,
                                    min_epsilon=0,
                                    target_update=2000,
                                    epochs=200000,
                                    batch_size=16,
                                    memory_size=8000)
        self.model.load_model()
    

    def move(self, environment):
        BaseGameModel.move(self, environment) 
        state = torch.tensor((environment.state())).float()
        vixod=self.model.action_choice(state=state, epsilon=0, model=self.model.model)

        return self.action_all[vixod]

    def reset(self):
        pass

"""print(f"Введите режим 0-если обучение, 1-если игра")
a = 1
if a==0:
    model=BaseGameModel("dqn_trainer", "dqn_trainer", "dqn_trainer")
    agent = Agent(
    env=model.prepare_training_environment(),
    file_name='snake_7.2',
    max_epsilon=1,
    min_epsilon=0,
    target_update=20,
    epochs=500000,
    batch_size=16,
    memory_size=8000)
    agent.load_model()
    agent.train()
if a==1:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)"""
    