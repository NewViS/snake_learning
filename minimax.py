import random
import copy
import time
from run import Run
from game import Game
from constants import Constants
from base_game_model import BaseGameModel

ITER = 300
iterate = 0
envs_q = []

class MiniMax(BaseGameModel):

    def __init__(self):
        BaseGameModel.__init__(self, "Monte Carlo", "monte_carlo", "mc")

    # def move(self, environment):
    #     possible_actions = environment.possible_actions_for_current_action(environment.snake_action)
    #     runs = []
    #     for run_index in range(0, self.runs):
    #         action = random.choice(possible_actions)
    #         new_environment = copy.deepcopy(environment)
    #         score = self._run(action, new_environment)
    #         runs.append(Run(action,score))
    #     return self._best_action_for_runs(runs)

    def move(self, environment):
        global iterate
        iterate = ITER
        act = self._run(copy.deepcopy(environment))
        return act

    def find_index(self, arr):
        inde = max(arr)
        count = 0
        for i in arr:
            if i == inde:   count+=1
        
        if count == 1:
            return arr.index(inde)
        else:
            k = random.randint(1, count)
            print(k)
            count = 0
            out = 0
            for i in arr:
                if i == inde:   
                    count+=1
                    if k == count:
                        return out
                out+=1

    def _run(self, environment):
        global iterate
        directs = [0, 0, 0]
        envs_q = []
        envs_q.append([environment, 0, -1, 1])      #env reward direction deep
        acts = environment.possible_actions_for_current_action(environment.snake_action)
        while iterate > 0 and len(envs_q)>0:
            ENV = envs_q.pop(0)
            action = ENV[0].possible_actions_for_current_action(ENV[0].snake_action)
            for i in range(3):
                env = copy.deepcopy(ENV[0])
                st, rew, done = env.full_step(action[i])
                if not(done):
                    envs_q.append([env, rew+ENV[1], ENV[2] if (iterate<(ITER-2)) else i, ENV[3]+1])
                    
                directs[ENV[2] if (iterate<(ITER-2)) else i] = (rew+ENV[1])/(ENV[3]+1)
                iterate -= 1
                
                
                
        print(directs)
        #time.sleep(20)
        
        print("\n")
        #return acts[directs.index(max(directs))]
        return acts[self.find_index(directs)]


    # def _run(self, action, environment):
    #     score = self._random_gameplay(environment, action)
    #     return score

    # def _random_gameplay(self, environment, action):
    #     new_action = action
    #     next_state, reward, terminal = environment.full_step(new_action)
    #     while terminal:
    #         print(next_state)
    #         environment.eat_fruit_if_possible()
    #         new_action = random.choice(environment.possible_actions_for_current_action(environment.snake_action))
    #         next_state, reward, terminal = environment.full_step(new_action)
    #     return environment.reward()

while True:
    Game(game_model=MiniMax(),
        fps=Constants.FPS,
        pixel_size=Constants.PIXEL_SIZE,
        screen_width=Constants.SCREEN_WIDTH,
        screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
        navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)