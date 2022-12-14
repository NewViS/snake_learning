import random
import copy
import time
from run import Run
from game import Game
from constants import Constants
from base_game_model import BaseGameModel

ITER = 300
act = []
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

    def move(self, environment):                           #correct move
        global iterate
        iterate = ITER
        act = self._run2(copy.deepcopy(environment))
        return act

    # def move(self, environment):
    #     global act
    #     if environment.Terminal:    
    #         print(act)
    #         act = []
    #     if len(act) == 0:
    #         act = self._run3(copy.deepcopy(environment))
    #     if len(act) != 0:    
    #         # print(act)
    #         return act.pop(0)
        


    def find_index(self, arr):
        inde = max(arr)
        count = 0
        for i in arr:
            if i == inde:   count+=1
        
        if count == 1:
            return arr.index(inde)
        else:
            k = random.randint(1, count)
            # print(k)
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
                
                
                
        # print(directs)
        # #time.sleep(20)
        
        # print("\n")
        #return acts[directs.index(max(directs))]
        return acts[self.find_index(directs)]

    def _run2(self, environment):
        directs = [0, 0, 0]
        acts = environment.possible_actions_for_current_action(environment.snake_action)
        for i in range(3):
            score = [0, 0]
            env = copy.deepcopy(environment)
            st, rew, done, eat = env.full_step_eat(acts[i])
            score = [rew, 1]
            # directs[i] = score[0]/score[1]
            if not(done) and not(eat):   
                for j in range(30):
                    ENV = copy.deepcopy(env)
                    # score = [0, 0]
                    for k in range(50):
                        
                        st, rew, done, eat = ENV.full_step_eat(random.choice(ENV.possible_actions_for_current_action(ENV.snake_action)))
                        score[0]+=rew
                        score[1]+=1
                        # print(score)
                        if done or eat:
                            break
            
            directs[i] = score[0]/score[1]
            # print(score, directs)
        # print(directs)           
        return acts[self.find_index(directs)]

    def _run3(self, environment):
        directs = [0, 0, 0]
        runs = [[], [], []]
        # print(runs)
        run_scores = [0, 0, 0]
        acts = environment.possible_actions_for_current_action(environment.snake_action)
        for i in range(3):
            score = [0, 0]
            env = copy.deepcopy(environment)
            st, rew, done, feat = env.full_step_eat(acts[i])
            score = [rew, 1]
            runs[i] = [acts[i]]
            run_scores[i] = score[0]/score[1]
            
            if done or feat: 
                # print("CONTINUIED",done,feat)
                continue 
           
            for j in range(20):
                ENV = copy.deepcopy(env)
                pre_score = copy.deepcopy(score)
                pre_runs = [acts[i]]
                done2 = False
                for k in range(10):
                    pre_runs.append(random.choice(ENV.possible_actions_for_current_action(ENV.snake_action)))
                    st, rew, done2, feat2 = ENV.full_step_eat(pre_runs[-1])
                    pre_score[0]+=rew
                    pre_score[1]+=1
                    
                    if done2 or feat2:
                        break
                
                if (run_scores[i] < pre_score[0]/pre_score[1]) and not(done2):
                    
                    run_scores[i] = pre_score[0]/pre_score[1]
                    runs[i] = pre_runs
                

                    
            
        # print(runs)    
        # print(run_scores)    
        return runs[self.find_index(run_scores)]

while True:
    Game(game_model=MiniMax(),
        fps=Constants.FPS,
        pixel_size=Constants.PIXEL_SIZE,
        screen_width=Constants.SCREEN_WIDTH,
        screen_height=Constants.SCREEN_HEIGHT+Constants.NAVIGATION_BAR_HEIGHT,
        navigation_bar_height=Constants.NAVIGATION_BAR_HEIGHT)