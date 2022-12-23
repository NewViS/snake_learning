import numpy as np

class Tile():
    empty = " "
    snake = "x"
    fruit = "$"
    wall = "#"
    head = "%"

    @staticmethod
    def grayscale(tile):
        if tile == Tile.empty:
            return np.float32(0)#np.uint8(255)
        elif tile == Tile.fruit:
            return np.float32(0.75)#np.uint8(200)
        elif tile == Tile.snake:
            return np.float32(0.25)#np.uint8(75)
        elif tile == Tile.head:
            return np.float32(0.5)
        else:
            return np.float32(1.0)#np.uint8(0)

if __name__ == "__main__":
    print('Cell = 10, fps = 6, bot_name = Hamilton')
    print('Cell = 15, fps = 9, bot_name = Monte Carlo')