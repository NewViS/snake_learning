
class Action:

    left = (-1, 0)
    up = (0, -1)
    right = (1, 0)
    down = (0, 1)

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1]

class Constants:

    SLITHERIN_NAME = "NeuroFreak"
    ICON_PATH = "./assets/snake_icon.png"
    FONT = "Arial"
    NAVIGATION_BAR_HEIGHT = 30
    FPS = 10
    PIXEL_SIZE = 40
    SCREEN_WIDTH = 880
    SCREEN_HEIGHT = 880
    SCREEN_DEPTH = 32
    ENV_HEIGHT = SCREEN_HEIGHT/PIXEL_SIZE
    ENV_WIDTH = SCREEN_WIDTH/PIXEL_SIZE

class Node:

    point = None
    previous_node = None
    action = None

    def __init__(self, point):
        self.point = point

    def __eq__(self, other):
        return self.point == other.point

    def __hash__(self):
        return hash(str(self.point.x)+str(self.point.y))

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self.x)+str(self.y))

class Tile():
    empty = " "
    snake = "x"
    fruit = "$"
    wall = "#"