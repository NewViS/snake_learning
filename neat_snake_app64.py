import random 
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
import os
import sys
import pickle
from matplotlib import pyplot as plt
import neat 
from math import log

generation_number = 0
best_foods = 0
best_fitness = 0
loop_punishment = 0.25
near_food_score = 0.2
far_food_score = 0.225
moved_score = 0.01
list_best_fitness = []
fig = plt.figure()
b=0
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename='trained/best_generation_instances.pickle'):
    obj = 0
    if os.path.getsize(filename)>0:
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
    return obj

# def load_object(filename='neat-checkpoint-9999'):
    # po = neat.Checkpointer.restore_checkpoint("neat-checkpoint-9999")



local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.ini')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


pop = neat.Population(config)
# pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1450')
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(1000))



plt.ion()
fig = plt.figure()
plt.title('Best fitness')
ax = fig.add_subplot(111)
line_best_fitness, = ax.plot(list_best_fitness, 'r-')

def save_best_generation_instance(instance, filename='trained/best_generation_instances.pickle'):
    # instances = []
    # if os.path.isfile(filename):
    #     instances = load_object(filename)
    # instances.append(instance)
    save_object(instance, filename)


def eval_fitness(genome, config):
    global best_foods
    # best_foods = 0
    global best_fitness 
    global loop_punishment 
    global near_food_score 
    global far_food_score 
    global moved_score
    global line_best_fitness
    global b
    # circle_check = [-1] * 16
    # circle_index = 0
    if b!=0:
        state = b.observation2()
    best_instance = None
    genome_number = 0

    global generation_number
    global pop
    action=Action.all()
    # action_66=7 
    
    
    if b!=0:
        b.set_fruit()
        b.set_snake()
    state = b.observation2()
    
    net = neat.nn.FeedForwardNetwork.create(genome,config)
    step_score = 1
    food_score = 0.0
    score=0
    hunger = 200
    fake_reward_pa=0
    countFrames = 0
    foods = 0
    time_out = 0
    hang_out = 0
    count_fl = 1
    reward = 0
    outputs = None

    for t in range(2000):
        countFrames += 1
        outputs = net.activate(state)
        # print(outputs)
        # print(outputs, max(outputs))
        direction = outputs.index(max(outputs))
        # old_action=action_66
        action_66=action[direction]

        hunger -= 1
        time_out += 1
        
        
        next_state, fake_reward_fu, done, feat = b.full_step_neat(action_66)
        
        

        if(time_out >= math.ceil(count_fl * 0.7 + 10)):
            reward -= 0.5/count_fl
            time_out = 0
        if(fake_reward_fu + fake_reward_pa == 0):
        
            if count_fl==1:
                size = 2
            else:
                size=count_fl
            reward += math.log(((size) + state[8])/((size)+ next_state[8])) / math.log(size)   

        if fake_reward_fu==1 :
            time_out = 0

            reward=1
            hunger=200        
            count_fl+=1
                
        if done or hunger==0:
            reward= -1
            score += reward
            break

        if(reward > 1):
            reward = 1
        elif (reward < -1):
            reward = -1
        
        fake_reward_pa=fake_reward_fu
        score += reward
        state=next_state

        best_foods = max(best_foods, count_fl)
        #print("qq7")
    
    # score=food_score-(step_score/100)
    
    
    #print(outputs)
    if not best_instance or score > best_fitness:
        best_instance = {
            'num_generation': generation_number,
            'fitness': score,
            'score': score,
            'genome': genome,
            'net': net,
        }
        save_best_generation_instance(best_instance)
    # best_foods = max(best_foods, count_fl)
    best_fitness = max(best_fitness, score)
    # if debuggin:
    # print(f"Generation {generation_number} \tGenome {genome_number} \tFoods {food_score} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
    genome_number += 1
    return score


# save_best_generation_instance(best_instance)
generation_number += 1

if generation_number % 200 == 0:
    # print("2222222222222222222222222222222222222")
    save_object(pop, 'trained/population.dat')
    print("Exporting population")
    # export population
    # save_object(pop,'population.dat')
    # export population
# print("3333333333333333333333333333333333333333333333333333")
# global list_best_fitness
# global fig
list_best_fitness.append(best_fitness)
line_best_fitness.set_ydata(np.array(list_best_fitness))
line_best_fitness.set_xdata(list(range(len(list_best_fitness))))
plt.xlim(0, len(list_best_fitness)-1)
plt.ylim(0, max(list_best_fitness)+0.5)
fig.canvas.draw()
fig.canvas.flush_events()
# print("444444444444444444444444444444444444444444444444444")
def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    global best_foods
    best_foods = 0
    for genome_id, genome in genomes:
        
        genome.fitness = eval_fitness( genome, config)
    print("111111111111111111111111111111111111111")
    print(best_foods)
    print("111111111111111111111111111111111111111")



class NEAT_trainer(BaseGameModel):
    global pop
    
    global list_best_fitness
    def __init__(self):
        BaseGameModel.__init__(self, "neat_trainer", "neat_trainer", "neat_trainer")
    def move(self, env):
        BaseGameModel.move(self, env)
        global b 
        global pop
        global list_best_fitness
        list_best_fitness = []
         # Returns a tuple of line objects, thus the comma
        b=env
        
        best_genome = pop.run(eval_genomes, n=10000)

class NEAT_play(BaseGameModel):
    state_size=10
    action_size=4
    pth_path= 'checkpoint.pth'
    action_all=Action.all()
    def __init__(self,):
        BaseGameModel.__init__(self, "neat", "dqn", "dqn")
        # self.model = DQN(19, 4).to("cpu")
        # load_model(self.model)
        g = load_object()
        # g=[(1, g)]
        # print(g)
        self.net = neat.nn.FeedForwardNetwork.create(g['genome'], config)
        

    def move(self, environment):
        BaseGameModel.move(self, environment)
        outputs = self.net.activate(environment.observation2())
        print(outputs)
        direction = outputs.index(max(outputs))
        print(direction)
        # state = environment.observation()
        # state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        # with torch.no_grad():
        #     # t.max(1) will return largest column value of each row.
        #     # second column on max result is index of where max element was
        #     # found, so we pick action with the larger expected reward.
        #     vixod = self.model(state).max(1)[1].view(1, 1)
        # #vixod=self.model.select_action(state=state, epsilon=0, model=self.model.model)
        # print(vixod)
        return Action.all()[direction]


print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    
    model=NEAT_trainer()
    model.move(model.prepare_training_environment())
  
if a==1:
    while True:
                Game(game_model=NEAT_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)