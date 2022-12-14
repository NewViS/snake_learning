import random
import copy
import time
from run import Run
from collections import OrderedDict
from game import Game
from constants import Constants
from base_game_model import BaseGameModel

ITER = 300
iterate = 0
envs_q = []

class MonteCarloSolver(BaseGameModel):

    def __init__(self, runs=100):
        BaseGameModel.__init__(self, "Monte Carlo", "monte_carlo", "mc")
        self.runs = runs

    def move(self, environment):
        possible_actions = environment.possible_actions_for_current_action(environment.snake_action)
        runs = []
        for run_index in range(0, self.runs):
            action = random.choice(possible_actions)
            new_environment = copy.deepcopy(environment)
            score = self._run(action, new_environment)
            runs.append(Run(action,score))
        return self._best_action_for_runs(runs)

    def _run(self, action, environment):
        score = self._random_gameplay(environment, action)
        return score

    def _random_gameplay(self, environment, action):
        new_action = action
        while environment.step(new_action):
            environment.eat_fruit_if_possible()
            new_action = random.choice(environment.possible_actions_for_current_action(environment.snake_action))
        return environment.reward()
      
    def _best_action_for_runs(self, runs):
        scores_for_actions = {}
        for run in runs:
            if run.action in scores_for_actions.keys():
                scores_for_action = scores_for_actions[run.action]
                scores_for_action.append(run.score)
                scores_for_actions[run.action] = scores_for_action
            else:
                scores_for_actions[run.action] = [run.score]

        average_scores_for_actions = {}
        for action in scores_for_actions.keys():
            average_score_for_action = sum(scores_for_actions[action]) / float(len(scores_for_actions[action]))
            average_scores_for_actions[action] = average_score_for_action
        sorted_average_scores_for_actions = OrderedDict(sorted(average_scores_for_actions.items(), key=lambda t: t[1]))
        return list(sorted_average_scores_for_actions.keys())[-1]

while True:
    Game(game_model=MonteCarloSolver(),
        fps=Constants.FPS,
        pixel_size=Constants.PIXEL_SIZE,
        screen_width=Constants.SCREEN_WIDTH,
        screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
        navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)