import pygame
from classes import Point, Constants
from pygame.locals import *
from os.path import join


class Game:

    pygame.init()

    def __init__(self, pixel_size, cell, bot_name):
        self.bot_name = bot_name
        self.pixel_size = pixel_size
        self.cell = cell
        self.head_up = pygame.image.load('src/Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('src/Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('src/Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('src/Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('src/Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('src/Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('src/Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('src/Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('src/Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('src/Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('src/Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('src/Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('src/Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('src/Graphics/body_bl.png').convert_alpha()

        self.apple = pygame.image.load('src/Graphics/apple.png').convert_alpha()

        self.font = pygame.font.Font(join("src/Font/magic.TTF"), 25)

        self.action_bot = 'Right'
        self.action = 'Right'
        self.disp = pygame.display.set_mode((1800,1010))
        pygame.display.set_caption("NeuroFreak")
        icon = pygame.image.load("src/Assets/snake_icon.png")
        pygame.display.set_icon(icon)
        self.disp.fill((175,215,70))
        self.surface_human = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell))
        self.surface_bot = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell))
        self.head_human = None
        self.tail_human = None
        self.head_bot = None
        self.tail_bot = None
        self.score_human = 1
        self.score_bot = 1
        if self.cell%2 == 0:
            self.points_fruit_human = [Point((self.cell-3), (self.cell//2-1))]
            self.points_fruit_bot = [Point((self.cell-3), (self.cell//2-1))]
        else:
            self.points_fruit_human = [Point((self.cell-3), ((self.cell-1)//2))]
            self.points_fruit_bot = [Point((self.cell-3), ((self.cell-1)//2))]
        if self.cell%2 == 0:
            self.points_snake_human = [Point(2, (self.cell//2-1))]
            self.points_snake_bot = [Point(2, (self.cell//2-1))]
        else: 
            self.points_snake_human = [Point(2, ((self.cell-1)//2))]
            self.points_snake_bot = [Point(2, (self.cell//2-1))]

    def draw_grass_human(self):
        grass_color1 = (167,209,61)
        grass_color2 = (175,215,70)
        for row in range(self.cell):
            if row % 2 == 0: 
                for col in range(self.cell):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface_human, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface_human, grass_color2, grass_rect)
            else:
                for col in range(self.cell):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface_human, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface_human, grass_color2, grass_rect)
        pygame.draw.rect(self.surface_human, (0,0,0), (0, 0, self.cell * self.pixel_size, self.cell * self.pixel_size), 3)
    
    def draw_grass_bot(self):
        grass_color1 = (167,209,61)
        grass_color2 = (175,215,70)
        for row in range(self.cell):
            if row % 2 == 0: 
                for col in range(self.cell):
                    if col % 2 == 0:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface_bot, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * self.pixel_size, row * self.pixel_size, self.pixel_size, self.pixel_size)
                        pygame.draw.rect(self.surface_bot, grass_color2, grass_rect)
            else:
                for col in range(self.cell):
                    if col % 2 != 0:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface_bot, grass_color1, grass_rect)
                    else:
                        grass_rect = pygame.Rect(col * 40, row * 40, 40, 40)
                        pygame.draw.rect(self.surface_bot, grass_color2, grass_rect)
        pygame.draw.rect(self.surface_bot, (0,0,0), (0, 0, self.cell * self.pixel_size, self.cell * self.pixel_size), 3)

    def draw_fruit_human(self):
        fruit_rect = pygame.Rect(int(self.points_fruit_human[0].x*self.pixel_size),int(self.points_fruit_human[0].y*self.pixel_size), self.pixel_size, self.pixel_size)
        self.surface_human.blit(self.apple,fruit_rect)

    
    def draw_fruit_bot(self):
        fruit_rect1 = pygame.Rect(int(self.points_fruit_bot[0].x*self.pixel_size),int(self.points_fruit_bot[0].y*self.pixel_size), self.pixel_size, self.pixel_size)
        self.surface_bot.blit(self.apple,fruit_rect1)

    def draw_snake_human(self):
        self.update_head_graphics_human()
        self.update_tail_graphics_human()

        for index,point in enumerate(self.points_snake_human):
            x_pos = point.x * self.pixel_size
            y_pos = point.y * self.pixel_size
            block_rect = pygame.Rect(x_pos,y_pos,self.pixel_size,self.pixel_size)
            if index == 0:
                self.surface_human.blit(self.head_human,block_rect)
            elif index == len(self.points_snake_human) - 1 and len(self.points_snake_human)>2:
                self.surface_human.blit(self.tail_human,block_rect)
            elif len(self.points_snake_human) > 2:
                previous_block = self.points_snake_human[index + 1] - point
                next_block = self.points_snake_human[index - 1] - point
                if previous_block.x == next_block.x:
                    self.surface_human.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    self.surface_human.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        self.surface_human.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        self.surface_human.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        self.surface_human.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        self.surface_human.blit(self.body_br,block_rect)
            else:
                if self.head_human == self.head_up or self.head_human == self.head_down:
                    self.surface_human.blit(self.body_vertical,block_rect)
                else:
                    self.surface_human.blit(self.body_horizontal,block_rect)

    def draw_snake_bot(self):
        self.update_head_graphics_bot()
        self.update_tail_graphics_bot()

        for index,point in enumerate(self.points_snake_bot):
            x_pos = point.x * self.pixel_size
            y_pos = point.y * self.pixel_size
            block_rect = pygame.Rect(x_pos,y_pos,self.pixel_size,self.pixel_size)
            if index == 0:
                self.surface_bot.blit(self.head_bot,block_rect)
            elif index == len(self.points_snake_bot) - 1 and len(self.points_snake_bot)>2:
                self.surface_bot.blit(self.tail_bot,block_rect)
            elif len(self.points_snake_bot) > 2:
                previous_block = self.points_snake_bot[index + 1] - point
                next_block = self.points_snake_bot[index - 1] - point
                if previous_block.x == next_block.x:
                    self.surface_bot.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    self.surface_bot.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        self.surface_bot.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        self.surface_bot.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        self.surface_bot.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        self.surface_bot.blit(self.body_br,block_rect)
            else:
                if self.head_bot == self.head_up or self.head_bot == self.head_down:
                    self.surface_bot.blit(self.body_vertical,block_rect)
                else:
                    self.surface_bot.blit(self.body_horizontal,block_rect)

    def update_head_graphics_human(self):
        if len(self.points_snake_human) > 1:
            head_relation_human = self.points_snake_human[1] - self.points_snake_human[0]
            if head_relation_human == Point(1,0): self.head_human = self.head_left
            elif head_relation_human == Point(-1,0): self.head_human = self.head_right
            elif head_relation_human == Point(0,1): self.head_human = self.head_up
            elif head_relation_human == Point(0,-1): self.head_human = self.head_down
        else:
            if self.action == 'Left': self.head_human = self.head_left
            elif self.action == 'Right': self.head_human = self.head_right
            elif self.action == 'Up': self.head_human = self.head_up
            elif self.action == 'Down': self.head_human = self.head_down

    def update_head_graphics_bot(self):
        if len(self.points_snake_bot) > 1:
            head_relation_bot = self.points_snake_bot[1] - self.points_snake_bot[0]
            if head_relation_bot == Point(1,0): self.head_bot = self.head_left
            elif head_relation_bot == Point(-1,0): self.head_bot = self.head_right
            elif head_relation_bot == Point(0,1): self.head_bot = self.head_up
            elif head_relation_bot == Point(0,-1): self.head_bot = self.head_down
        else:
            if self.action_bot == 'Left': self.head_bot = self.head_left
            elif self.action_bot == 'Right': self.head_bot = self.head_right
            elif self.action_bot == 'Up': self.head_bot = self.head_up
            elif self.action_bot == 'Down': self.head_bot = self.head_down

    def update_tail_graphics_human(self):
        if len(self.points_snake_human) > 2:
            tail_relation_human = self.points_snake_human[-2] - self.points_snake_human[-1]
            if tail_relation_human == Point(1,0): self.tail_human = self.tail_left
            elif tail_relation_human == Point(-1,0): self.tail_human = self.tail_right
            elif tail_relation_human == Point(0,1): self.tail_human = self.tail_up
            elif tail_relation_human == Point(0,-1): self.tail_human = self.tail_down
        else:
            self.tail_human = None

    def update_tail_graphics_bot(self):
        if len(self.points_snake_bot) > 2:
            tail_relation_bot = self.points_snake_bot[-2] - self.points_snake_bot[-1]
            if tail_relation_bot == Point(1,0): self.tail_bot = self.tail_left
            elif tail_relation_bot == Point(-1,0): self.tail_bot = self.tail_right
            elif tail_relation_bot == Point(0,1): self.tail_bot = self.tail_up
            elif tail_relation_bot == Point(0,-1): self.tail_bot = self.tail_down
        else:
            self.tail_bot = None

    def draw_score_human(self):
        score_text = str(self.score_human) + '  Human'
        score_surface = self.font.render(score_text,True,(56,74,12))
        score_x = int(450-self.cell*self.pixel_size/2+40)
        score_y = int(505-self.cell*self.pixel_size/2)
        score_rect = score_surface.get_rect(bottomleft = (score_x,score_y))
        apple_rect = self.apple.get_rect(midright = (score_rect.left,score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left,apple_rect.top, 140 ,apple_rect.height)

        pygame.draw.rect(self.disp,(167,209,61),bg_rect)
        self.disp.blit(score_surface,score_rect)
        self.disp.blit(self.apple,apple_rect)
        pygame.draw.rect(self.disp,(56,74,12),bg_rect,2)

    def draw_score_bot(self):
        score_text = str(self.score_bot) + '  ' + self.bot_name
        score_surface = self.font.render(score_text,True,(56,74,12))
        score_x = int(1350-self.cell*self.pixel_size/2+40)
        score_y = int(505-self.cell*self.pixel_size/2)
        score_rect = score_surface.get_rect(bottomleft = (score_x,score_y))
        apple_rect = self.apple.get_rect(midright = (score_rect.left,score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left,apple_rect.top, 195 ,apple_rect.height)

        pygame.draw.rect(self.disp,(167,209,61),bg_rect)
        self.disp.blit(score_surface,score_rect)
        self.disp.blit(self.apple,apple_rect)
        pygame.draw.rect(self.disp,(56,74,12),bg_rect,2)

    def draw_elements_human(self):
        self.draw_grass_human()
        self.draw_fruit_human()
        self.draw_snake_human()
        self.draw_score_human()
        self.disp.blit(self.surface_human, (450-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))

    def draw_elements_bot(self):
        self.draw_grass_bot()
        self.draw_fruit_bot()
        self.draw_snake_bot()
        self.draw_score_bot()
        self.disp.blit(self.surface_bot, (1350-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))

    def pause_human_win(self):
        pause = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Вы выиграли'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (self.pixel_size*self.cell/2,self.pixel_size*self.cell/2))   
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (450-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))

    def pause_human_lose(self):
        pause = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Вы проиграли'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (self.pixel_size*self.cell/2,self.pixel_size*self.cell/2))   
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (450-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))

    def pause_bot_win(self):
        pause = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Вы выиграли'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (self.pixel_size*self.cell/2,self.pixel_size*self.cell/2))   
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (1350-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))

    def pause_bot_lose(self):
        pause = pygame.Surface((self.pixel_size*self.cell, self.pixel_size*self.cell), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Вы проиграли'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (self.pixel_size*self.cell/2,self.pixel_size*self.cell/2))   
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (1350-self.cell*self.pixel_size/2, 505-self.cell*self.pixel_size/2))
    
    def end(self):
        pause = pygame.Surface((1800,1010), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Нажмите любую кнопку, чтобы вернуться в меню'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (900,505))  
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (0,0))
    def pause(self):
        pause = pygame.Surface((1800,1010), pygame.SRCALPHA)   
        pause.fill((0, 0, 0, 85))
        text = 'Пауза'
        text_surface = self.font.render(text,True,(56,74,12))
        text_rect = text_surface.get_rect(center = (900,505))  
        pause.blit(text_surface, text_rect)
        self.disp.blit(pause, (0,0))

    def _display(self):
        pygame.display.flip()
        pygame.display.update()
