import argparse
import random
from constants import Constants
from game import Game
from minimax import MiniMax
from DQN import DQN,DQN_trainer
from neat import  NEAT, NEAT_Trainer
from human import Human
from hamilton import Hamilton


solvers = [
           NEAT(),
           Human(),
           MiniMax(),
           Hamilton(),
           DQN()
           ]

trainers = [
            NEAT_Trainer(),
            DQN_trainer()
            ]

game_models = solvers + trainers


def args():
    parser = argparse.ArgumentParser()
    for game_model in game_models:
        parser.add_argument("-"+game_model.abbreviation, "--"+game_model.short_name,
                            help=game_model.long_name,
                            action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    selected_game_model = random.choice(solvers)
    args = args()
    for game_model in game_models:
        if game_model.short_name in args and vars(args)[game_model.short_name]:
            selected_game_model = game_model
            print(str(selected_game_model))
    if selected_game_model in trainers:
        selected_game_model.move(selected_game_model.prepare_training_environment())
        print(selected_game_model)
    else:
        while True:
            Game(game_model=selected_game_model,
                fps=Constants.FPS,
                pixel_size=Constants.PIXEL_SIZE,
                screen_width=Constants.SCREEN_WIDTH,
                screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
                navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)
