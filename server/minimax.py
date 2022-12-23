import random
import copy
#from game import Game
from constants import Constants
from base_game_model import BaseGameModel


class MiniMax(BaseGameModel):

    ITER = 300
    act = []
    iterate = 0
    #envs_q = []

    def move(self, environment):                           #correct move
        global iterate
        iterate = self.ITER
        self.act = self._run2(copy.deepcopy(environment))
        return self.act

    def find_index(self, arr):
        inde = max(arr)
        count = 0
        for i in arr:
            if i == inde:   count+=1
        
        if count == 1:
            return arr.index(inde)
        else:
            k = random.randint(1, count)
            count = 0
            out = 0
            for i in arr:
                if i == inde:   
                    count+=1
                    if k == count:
                        return out
                out+=1

    def _run2(self, environment):
        directs = [0, 0, 0]
        acts = environment.possible_actions_for_current_action(environment.snake_action)
        for i in range(3):
            score = [0, 0]
            env = copy.deepcopy(environment)
            st, rew, done, eat = env.full_step_neat(acts[i])
            score = [rew, 1]
            # directs[i] = score[0]/score[1]
            if not(done) and not(eat):   
                for j in range(2):
                    ENV = copy.deepcopy(env)
                    for k in range(5):
                        
                        st, rew, done, eat = ENV.full_step_neat(random.choice(ENV.possible_actions_for_current_action(ENV.snake_action)))
                        score[0]+=rew
                        score[1]+=1
                        if done or eat:
                            break
            
            directs[i] = score[0]/score[1]        
        return acts[self.find_index(directs)]
    def reset(self):
        self.ITER = 300
        self.act = []
        self.iterate = 0