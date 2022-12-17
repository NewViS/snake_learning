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
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.ini')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


pop = neat.Population(config)

pop.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(100))



plt.ion()
fig = plt.figure()
plt.title('Best fitness')
ax = fig.add_subplot(111)
line_best_fitness, = ax.plot(list_best_fitness, 'r-')

def save_best_generation_instance(instance, filename='trained/best_generation_instances.pickle'):
    instances = []
    if os.path.isfile(filename):
        instances = load_object(filename)
    instances.append(instance)
    save_object(instances, filename)


def eval_fitness(genomes,config):
    global best_foods
    global best_fitness 
    global loop_punishment 
    global near_food_score 
    global far_food_score 
    global moved_score
    global line_best_fitness
    global b
    circle_check = [-1] * 16
    circle_index = 0
    if b!=0:
        state = b.observation()
    best_instance = None
    genome_number = 0

    global generation_number
    global pop
    action=Action.all()
    action_66=7 
    print("00000000000000000000000000000000000000000")
    for g_id,g in genomes:
        if b!=0:
            b.set_fruit()
            b.set_snake()
        state = b.observation()
    
        net = neat.nn.FeedForwardNetwork.create(g,config)
        dx = 1
        dy = 0
        score = 0.0
        mindist = state[8]
        hunger = 100
        fake_reward_pa=0
        error = 0
        countFrames = 0
        circle_check = [-1] * 16
        circle_index = 0
        pastPoints = set()
        action_66=7 
        foods = 0

        for t in range(1500):
            #print("qq1")
            countFrames += 1
            outputs = net.activate(state)
            #print("qq2")
            direction = outputs.index(max(outputs))
            #print("qq3")
            old_action=action_66
            action_66=action[direction]

            hunger -= 1
            if countFrames>1:
                if Action.is_reverse(old_action, action_66) :
                    score -= 0.25
                if action_66 != old_action:
                    circle_check[circle_index % len(circle_check)] = direction
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
                        score -= loop_punishment
            #print("qq4")
            next_state, fake_reward_fu, done = b.full_step(action_66)
            #print("qq5")
                
            if  done:
                #print("qq6")
                break
            else:
                score += moved_score
            if fake_reward_fu> fake_reward_pa:
                score += 5
                hunger += 100
                foods += 1
                mindist=next_state[8]

            #if state[8]<next_state[8]:
            #    score += near_food_score
            #if state[8]>next_state[8]:
            #    score -= far_food_score
            
            if mindist<next_state[8]:
                score += near_food_score
                mindist = next_state[8]


            fake_reward_pa=fake_reward_fu
            state=next_state
            #print("qq7")
        g.fitness = score/100

        if not best_instance or g.fitness > best_fitness:
            best_instance = {
                'num_generation': generation_number,
                'fitness': g.fitness,
                'score': score,
                'genome': g,
                'net': net,
            }
        best_foods = max(best_foods, foods)
        best_fitness = max(best_fitness, g.fitness)
        # if debuggin:
        print(f"Generation {generation_number} \tGenome {genome_number} \tFoods {foods} \tBF {best_foods} \tFitness {g.fitness} \tBest fitness {best_fitness} \tScore {score}")
        genome_number += 1
    print("111111111111111111111111111111111111111")
    save_best_generation_instance(best_instance)
    generation_number += 1

    # if generation_number % 20 == 0:
    #     print("2222222222222222222222222222222222222")
    #     save_object(pop, 'trained/population.dat')
    #     print("Exporting population")
    #     # export population
    #     # save_object(pop,'population.dat')
    #     # export population
    print("3333333333333333333333333333333333333333333333333333")
    global list_best_fitness
    global fig
    list_best_fitness.append(best_fitness)
    line_best_fitness.set_ydata(np.array(list_best_fitness))
    line_best_fitness.set_xdata(list(range(len(list_best_fitness))))
    plt.xlim(0, len(list_best_fitness)-1)
    plt.ylim(0, max(list_best_fitness)+0.5)
    fig.canvas.draw()
    fig.canvas.flush_events()
    print("444444444444444444444444444444444444444444444444444")
def eval_genomes(_genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    net = []
    genomes = []
    

    for j, (i, genome) in enumerate(_genomes):
        genome.fitness = -1000
        n = neat.nn.FeedForwardNetwork.create(genome, config)
        net.append(n)
        genomes.append(genome)
    global b
    state = b.observation()
    dx = 1
    dy = 0
    score = 0.0
    mindist = state[8]
    hunger = 100
    fake_reward_pa=0
    error = 0
    countFrames = 0
    circle_check = [-1] * 16
    circle_index = 0
    pastPoints = set()
    action_66=7 
    foods = 0
    global best_foods
    global best_fitness 
    global loop_punishment 
    global near_food_score 
    global far_food_score 
    global moved_score
    global line_best_fitness
   
    circle_check = [-1] * 16
    circle_index = 0
    
    best_instance = None
    genome_number = 0
    
    global generation_number
    global pop
    action=Action.all()
    action_66=7 
        

    run = True

    for t in range(625):
        run = False
        countFrames += 1
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        outputs = net[genome_number].activate(state)
            #print("qq2")
        direction = outputs.index(max(outputs))
            #print("qq3")
        old_action=action_66
        action_66=action[direction]
        
        if countFrames>1:
            if Action.is_reverse(old_action, action_66) :
                score -= 0.25
            if action_66 != old_action:
                circle_check[circle_index % len(circle_check)] = direction
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
                    genomes[genome_number].fitness -= 5
        
        next_state, fake_reward_fu, done = b.full_step(action_66)
            #print("qq5")
        print("?????????????????????????????????????????????????????????????????????????????????????????????????")        
        if  done:
                #print("qq6")
            break
        print("?????????????????????????????????????????????????????????????????????????????????????????????????")
        if fake_reward_fu> fake_reward_pa:
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            genomes[genome_number].fitness += 35
            hunger += 100
            foods += 1
            mindist=next_state[8]

        if state[8]<next_state[8]:
            genomes[genome_number].fitness += 1.5
        if state[8]>next_state[8]:
            genomes[genome_number].fitness -= 1.5
            
        # if mindist<next_state[8]:
        #     score += near_food_score
        #     mindist = next_state[8]


        fake_reward_pa=fake_reward_fu
        state=next_state
        genome_number+=1

    #     # Check if snake go closer, then award if
    #     if curr <= prev:
    #         genomes[i].fitness += 1.5
    #     else:
    #         genomes[i].fitness -= 1.5
    #     # genomes[i].fitness += 0.5

    #     if apple == game.snake.pos[0]:
    #         genomes[i].fitness += 35

    #     # Check if Snake is going at the one place
    #     if game.snake.pos[0] in game.snake.path[:-1]:
    #         genomes[i].fitness -= 5

    #     if len(game.snake.path) > 625:
    #         game.run = False

    # for genome_id, genome in genomes:
        
    #     genome.fitness = eval_fitness( genome, config)



class NEAT_trainer(BaseGameModel):
    global pop
    
    global list_best_fitness
    def __init__(self):
        BaseGameModel.__init__(self, "neat_trainer", "neat_trainer", "neat_trainer")
    def move(self, env,):
        BaseGameModel.move(self, env)
        global b 
        global pop
        global list_best_fitness
        list_best_fitness = []
         # Returns a tuple of line objects, thus the comma
        b=env
        
        pop.run(eval_genomes, n=10000)



print(f"Введите режим 0-если обучение, 1-если игра")
a = int(input())
if a==0:
    
    model=NEAT_trainer()
    model.move(model.prepare_training_environment())
  
if a==1:
    while True:
                Game(game_model=DQN_play(),
                    fps=Constants.FPS,
                    pixel_size=Constants.PIXEL_SIZE,
                    screen_width=Constants.SCREEN_WIDTH,
                    screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                    navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)