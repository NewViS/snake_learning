
class Constants:

    SLITHERIN_NAME = "Snake"
    ICON_PATH = "./assets/snake_icon.png"
    FONT = "Arial"
    NAVIGATION_BAR_HEIGHT = 30
    FPS = 10
    PIXEL_SIZE = 25
    SCREEN_WIDTH = 300
    SCREEN_HEIGHT = 300
    SCREEN_DEPTH = 32
    ENV_HEIGHT = SCREEN_HEIGHT/PIXEL_SIZE
    ENV_WIDTH = SCREEN_WIDTH/PIXEL_SIZE

class Colour:

    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 150, 0)
    white = (255, 255, 255)
    gray = (211, 211, 211)

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        if key == 1:
            return self.y

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x,y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self.x)+str(self.y))