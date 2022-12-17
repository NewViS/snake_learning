import random
import math
import numpy as np
from collections import deque
from point import Point
from action import Action
from tile import Tile
from constants import Constants


class Environment:

    snake = []
    fruit = []
    wall = []

    snake_moves = 0
    snake_length = 1
    snake_action = None
    Terminal = None

    def __init__(self, width=Constants.ENV_WIDTH, height=Constants.ENV_HEIGHT):
        self.width = width
        self.height = height
        self.tiles = []
        self.frames = []
        for y in range(0, int(self.height)):
            self.tiles.append([])
            for x in range(0, int(self.width)):
                self.tiles[y].append(Tile.empty)

    def full_step(self, action):
        
        reward = 1 if self.eat_fruit_if_possible() else 0
        
        terminal = not self.step(action)
        
        if terminal:
            reward = -1
            
        self.terminal = terminal
        state = self.observation()
        return state, reward, terminal

    def step(self, action):
        action_all=Action.all()
        
        if Action.is_reverse(self.snake_action, action):
            action = self.snake_action
        
        self.snake_action = action
        
        head = self.snake[0]
        
        x, y = self.snake_action
        
        new = Point(x=(head.x + x),
                    y=(head.y + y))
        
        if new in self.snake:

            self.snake_length=1
            return False
        elif new in self.wall:
        
            self.snake_length=1
            return False
        else:

            
            self.snake_moves += 1
            
            self.snake.insert(0, new)
            
            self.tiles[new.y][new.x] = Tile.snake
            if len(self.snake) > self.snake_length:
                last = self.snake.pop()
                self.tiles[last.y][last.x] = Tile.empty
            
            
            return True

    def state(self):
        return np.asarray(self._frames())

    def reward(self):
        return self.snake_length

    def distance_from_fruit(self):
        head = self.snake[0]
        fruit = self.fruit[0]
        x_distance = abs(head.x - fruit.x)
        y_distance = abs(head.y - fruit.y)
        return math.hypot(x_distance, y_distance)

    def distance_from_up_wall(self):
        head = self.snake[0]
        y_distance = self.height
        return abs(head.y-y_distance)

    def distance_from_down_wall(self):
        head = self.snake[0]
        y_distance = 0
        return abs(head.y-y_distance)

    def distance_from_left_wall(self):
        head = self.snake[0]
        x_distance = 0
        return abs(head.x-x_distance)

    def distance_from_right_wall(self):
        head = self.snake[0]
        x_distance = self.width
        return abs(head.x-x_distance)

    def distance_from_ur_wall(self):
        distance= min(self.distance_from_up_wall(),self.distance_from_right_wall())
        return distance/math.cos(math.pi/4)

    def distance_from_ul_wall(self):
        distance= min(self.distance_from_up_wall(),self.distance_from_left_wall())
        return distance/math.cos(math.pi/4)

    def distance_from_dr_wall(self):
        distance= min(self.distance_from_down_wall(),self.distance_from_right_wall())
        return distance/math.cos(math.pi/4)

    def distance_from_dl_wall(self):
        distance= min(self.distance_from_down_wall(),self.distance_from_left_wall())
        return distance/math.cos(math.pi/4)

    def observation(self,):
        head = self.snake[0]
        distance_from_left_tail=0
        distance_from_right_tail=0
        distance_from_up_tail=0
        distance_from_down_tail=0
        distance_from_ul_tail= 0
        distance_from_ur_tail= 0
        distance_from_dl_tail= 0
        distance_from_dr_tail = 0
        for i in range(1,self.snake_length):

            minus_x=head.x - self.snake[i].x
            minus_y=head.y - self.snake[i].y
            if head.y==self.snake[i].y and minus_x < 0 and  distance_from_right_tail==0:
                distance_from_right_tail= minus_x
            if head.y==self.snake[i].y and minus_x > 0 and  distance_from_left_tail==0:
                distance_from_left_tail = minus_x
            if head.x==self.snake[i].x and minus_y < 0 and  distance_from_up_tail ==0:
                distance_from_up_tail = minus_y
            if head.x==self.snake[i].x and minus_y> 0 and  distance_from_down_tail==0:
                distance_from_down_tail = minus_y
            if abs(minus_x)==abs(minus_y):
                if minus_x >0 and minus_y >0 and distance_from_ur_tail ==0:
                    distance_from_ur_tail =math.hypot(minus_x, minus_y)
                if minus_x <0 and minus_y >0 and distance_from_ul_tail ==0:
                    distance_from_ul_tail =math.hypot(minus_x, minus_y)
                if minus_x >0 and minus_y <0 and distance_from_dr_tail ==0:
                    distance_from_dr_tail =math.hypot(minus_x, minus_y)
                if minus_x < 0 and minus_y < 0 and distance_from_dl_tail ==0:
                    distance_from_dl_tail =math.hypot(minus_x, minus_y)

        return [self.distance_from_up_wall(),
        self.distance_from_down_wall(),
        self.distance_from_left_wall(), 
        self.distance_from_right_wall(),
        self.distance_from_dr_wall(),
        self.distance_from_dl_wall(),
        self.distance_from_ur_wall(),
        self.distance_from_ul_wall(),
        self.distance_from_fruit(), 
        self._angle_from_fruit()+math.pi,
        abs(distance_from_right_tail),
        abs(distance_from_left_tail),
        abs(distance_from_up_tail),
        abs(distance_from_down_tail),
        abs(distance_from_ur_tail),
        abs(distance_from_ul_tail),
        abs(distance_from_dr_tail),
        abs(distance_from_dl_tail),
        self.snake_length]

    def possible_actions_for_current_action(self, current_action):
        actions = Action.all()
        reverse_action = (current_action[0] * -1, current_action[1] * -1)
        actions.remove(reverse_action)
        return actions

    def eat_fruit_if_possible(self):
        if self.fruit[0] == self.snake[0]:
            
            self.snake_length += 1
            
            self.snake_moves = 0
            if self._is_winning():
                return True
            self.set_fruit()
            return True
        return False

    def set_wall(self):
        for y in range(0, int(self.height)):
            for x in range(0, int(self.width)):
                if x == 0 or x == self.width-1 or y == 0 or y == self.height-1:
                    self.tiles[y][x] = Tile.wall
        self.wall = self._points_of(Tile.wall)
        return self.wall

    def set_fruit(self):
        self._clear_environment_for(Tile.fruit)
        random_position = self._random_available_position()
        self.tiles[random_position.x][random_position.y] = Tile.fruit
        self.fruit = self._points_of(Tile.fruit)
        
        return self.fruit

    def set_snake(self):
       
        self._clear_environment_for(Tile.snake)
        random_position = self._random_available_position()
        self.tiles[random_position.x][random_position.y] = Tile.snake
        self.snake = self._points_of(Tile.snake)
        self.snake_length = 1
        self.snake_moves = 0
        if self.snake_action is None:
            self.snake_action = random.choice(Action.all())
        
        return self.snake

    def print_path(self, path):
        environment_string = ""
        for y in range(0, self.height):
            environment_string += "\n"
            for x in range(0, self.width):
                tile = self.tiles[y][x]
                for p in path:
                    if tile == Tile.empty and p.point == Point(x, y):
                        tile = Action.description(p.action)
                environment_string += " " + tile + " "
        print (environment_string)

    def print_to_console(self):
        environment_string = ""
        for y in range(0, self.height):
            environment_string += "\n"
            for x in range(0, self.width):
                environment_string += " " + self.tiles[y][x] + " "
        print (environment_string)

    def _frame(self):
        grayscale = [[Tile.grayscale(tile) for tile in row] for row in self.tiles]
        return np.array(grayscale)

    def _frames(self):
        while len(self.frames) < Constants.FRAMES_TO_REMEMBER:
            self.frames.append(self._frame())
        return self.frames

    def _update_frames(self):
        self.frames.append(self._frame())
        while len(self.frames) > Constants.FRAMES_TO_REMEMBER:
            self.frames.pop(0)

    def is_in_fruitless_cycle(self):
        return self.snake_moves >= self._available_tiles_count()

    def _angle_from_fruit(self):
        snake = self.snake[0]
        fruit = self.fruit[0]
        angle = math.atan2(fruit.y - snake.y, fruit.x - snake.x)
        adjusted_angles = Action.adjusted_angles(self.snake_action)
        adjusted_angle_cw = angle + adjusted_angles[0]
        adjusted_angle_ccw = angle - adjusted_angles[1]
        if abs(adjusted_angle_cw) < abs(adjusted_angle_ccw):
            return adjusted_angle_cw
        else:
            return adjusted_angle_ccw

    def _is_point_accessible(self, point):
        return int(self.tiles[point.y][point.x] == Tile.empty
                   or self.tiles[point.y][point.x] == Tile.fruit)

    def _points_of(self, environment_object):
        points = []
        for y in range(0, int(self.height)):
            for x in range(0, int(self.width)):
                tile = self.tiles[y][x]
                if tile == environment_object:
                    points.append(Point(x, y))
        return points

    def _clear_environment_for(self, environment_object):
        points_to_clear = self._points_of(environment_object)
        for point in points_to_clear:
            self.tiles[point.y][point.x] = Tile.empty

    def _random_available_position(self):
        tile = None
        while tile is None or tile is not Tile.empty:
            random_x = random.randint(0, self.height-1)
            random_y = random.randint(0, self.width-1)
            tile = self.tiles[random_x][random_y]
        return Point(random_x, random_y)

    def _available_tiles_count(self):
        return (self.width-2) * (self.height-2)

    def _is_winning(self):
        return self.reward() == self._available_tiles_count()