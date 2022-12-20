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

HIGH_SCORE = 0

def main(genomes, config):
    nets = []
    ge = []
    snakes = []
    apples = []
    move_count = []
    game_model= BaseGameModel("neat_trainer", "neat_trainer", "neat_trainer")
    env=game_model.prepare_training_environment()
    global HIGH_SCORE
    block_size = 40
    

    for _, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(env.set_snake())
        apples.append(env.set_fruit())
        ge.append(g)
        move_count.append(0)
    action=Action.all()
    while True:
        print(HIGH_SCORE)

        # Give the outputs to the NN
        for i, snake in enumerate(snakes):
            output = nets[i].activate(env.observation())
            
            action_66=action[output.index(max(output))]
            env.snake=snake
            env.snake_length=len(snake)
            next_state, fake_reward_fu, done = env.full_step(action_66)        
            move_count[i] += 1
            HIGH_SCORE = max(HIGH_SCORE, env.snake_length)
        # Eat apple
            if fake_reward_fu:
                ge[i].fitness += 20
                move_count[i] =  0
            # Collision
            if done  or move_count[i] >= 120:
                ge[i].fitness -= (10 + move_count[i]//10)
                snakes.pop(i)
                nets.pop(i)
                ge.pop(i)
                apples.pop(i)
                move_count.pop(i)
        if len(snakes) == 0:
            break
        


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50, ))
    winner = p.run(main, 10000)

    


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)