import pygame


import neat
import os
import math
import pickle
from base_game_model import BaseGameModel
from action import Action
from constants import Constants
global show_game
show_game = False
global block_size
block_size = 20


game_model= BaseGameModel("neat_trainer", "neat_trainer", "neat_trainer")
action=Action.all()
env=game_model.prepare_training_environment()

def run_game(genome, config):
    global show_game
    global block_size
    global action
    global env
    env.set_snake()
    env.set_fruit()
    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    
    frames = 0
    max_frames = 80
    fruits = 0
    run = True
    fruits_max = 0
    fake_reward_pa=0
    count_fl = 1
    reward = 0
    score = 0
    observ= env.observation()
    time_out=0
    hunger = 200
    while run:
        # reward+=0.1
        time_out+=1
        frames +=1
        hunger -= 1  
        
        decisions = agent.activate(observ)
        if max(decisions) > 0.5:
            decision = decisions.index(max(decisions))
        else:
            decision = -1
        #snake.move()
        next_observ, fake_reward_fu, done = env.full_step(action[decision])
        # if next_observ[8]< observ[8]:
        #     reward+=1
        # elif next_observ[8]> observ[8]:
        #     reward-=1
        # if fake_reward_fu:
        #     max_frames += 80
        #     fruits += 1

        # if done:
        #     if fruits> fruits_max:
        #         fruits_max = fruits
        #     run = False
        # if(time_out >= math.ceil(count_fl * 0.7 + 10)):
        #     reward -= 0.5/count_fl
        #     time_out = 0
        # if(fake_reward_fu + fake_reward_pa == 0):
                
        #     if count_fl==1:
        #         size = 2
        #     else:
        #         size=count_fl
        #         reward += math.log(((size) + observ[8])/((size)+ next_observ[8])) / math.log(size)   
				
        if fake_reward_fu==1 :
            time_out = 0
                   
            reward+=1
            hunger=200        
            count_fl+=1

        if frames>= 100 and count_fl<=5:
            reward-=10
                    # if (count_fl> 10):
                    #     hang_out = math.ceil(0.4 * count_fl) + 2
                    # else:
                    #     hang_out = 6

        if done or hunger==0:
            
            run = False
			
            # if hunger<0:
            #     reward -= 1     
            # if mindist>next_state[8]:
            #     reward += 0.5
            #     mindist = next_state[8]
        # if(reward > 1):
        #     reward = 1
        # elif (reward < -1):
        #     reward = -1
        observ  =  next_observ
        score+= reward
    
    return reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = run_game(genome, config)


TRAINNING = True



local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.ini')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

if TRAINNING:
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('old-best-neat')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))

    show_game = False
    winner = p.run(eval_genomes, 50000)
else:
    with open('best-snake', 'rb') as snake:
        winner = pickle.load(snake)
    show_game = True
    run_game(winner, config)
