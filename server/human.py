from base_game_model import BaseGameModel
from action import Action
from constants import Constants
from pygame.locals import *


class Human(BaseGameModel):

    action = None

    def move(self, environment):
        BaseGameModel.move(self, environment)
        if self.action is None:
            return environment.snake_action
        backward_action = self.action[0] == environment.snake_action[0] * -1 or self.action[1] == environment.snake_action[1] * -1
        return environment.snake_action if backward_action else self.action
    
    def reset(self):
        self.action = None

