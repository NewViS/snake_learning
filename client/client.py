import socketio
from classes import Constants, Point
from game import Game
from os.path import dirname, join
import pygame, sys
from pygame.locals import *
from math import cos, radians

sio = socketio.Client()

@sio.event
def play():
    while True:
        pygame.time.Clock().tick(fps)
        action = [0, 0]
        for even in pygame.event.get():
            if even.type == QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit()
            elif even.type == KEYDOWN:
                if even.key == K_SPACE:
                    pygame.event.clear()
                    game.pause()
                    game._display()
                    ev= pygame.event.wait()
                    if ev.type == KEYDOWN:
                        if ev.key == K_SPACE:
                            game.disp.fill((175,215,70))
                            pass
                        elif ev.key == K_ESCAPE:
                            mymenu.runm()
                elif even.key == K_UP:
                    action = [0, -1]
                    if game.action != 'Down':
                        game.action = 'Up'
                elif even.key == K_DOWN:
                    action = [0, 1]
                    if game.action != 'Up':
                        game.action = 'Down'
                elif even.key == K_LEFT:
                    action = [-1, 0]
                    if game.action != 'Right':
                        game.action = 'Left'
                elif even.key == K_RIGHT:
                    action = [1, 0]
                    if game.action != 'Left':
                        game.action = 'Right'
        sio.emit("game", action)  

@sio.event
def get_both_env(arg):
    fruit_snake_score_human = [[], [], []]
    fruit_snake_score_bot = [[], [], [], []]
    fruit_snake_score_human[0].append(Point(arg[0][0][0][0]-1, arg[0][0][0][1]-1))
    for i in range(len(arg[0][1])):
        fruit_snake_score_human[1].append(Point(arg[0][1][i][0]-1, arg[0][1][i][1]-1))
    fruit_snake_score_human[2].append(arg[0][2])
    game.points_fruit_human = fruit_snake_score_human[0]
    game.points_snake_human = fruit_snake_score_human[1]
    game.score_human = fruit_snake_score_human[2][0][0]

    fruit_snake_score_bot[0].append(Point(arg[1][0][0][0]-1, arg[1][0][0][1]-1))
    for i in range(len(arg[1][1])):
        fruit_snake_score_bot[1].append(Point(arg[1][1][i][0]-1, arg[1][1][i][1]-1))
    fruit_snake_score_bot[2].append(arg[1][2])
    fruit_snake_score_bot[3].append(arg[1][3])
 

    game.points_fruit_bot = fruit_snake_score_bot[0]
    game.points_snake_bot = fruit_snake_score_bot[1] 
    game.score_bot = fruit_snake_score_bot[2][0][0]
    if fruit_snake_score_bot[3][0][0][0] == 1 and fruit_snake_score_bot[3][0][0][1] == 0:
        game.action_bot = 'Right'
    elif fruit_snake_score_bot[3][0][0][0] == -1 and fruit_snake_score_bot[3][0][0][1] == 0:
        game.action_bot = 'Left'
    elif fruit_snake_score_bot[3][0][0][0] == 0 and fruit_snake_score_bot[3][0][0][1] == 1:
        game.action_bot = 'Down'
    elif fruit_snake_score_bot[3][0][0][0]== 0 and fruit_snake_score_bot[3][0][0][1] == -1:
        game.action_bot = 'Up'
    game.draw_elements_human()
    game.draw_elements_bot()
    game._display()


@sio.event
def get_env_human(arg):
    fruit_snake_score_human = [[], [], []]
    fruit_snake_score_human[0].append(Point(arg[0][0][0]-1, arg[0][0][1]-1))
    for i in range(len(arg[1])):
        fruit_snake_score_human[1].append(Point(arg[1][i][0]-1, arg[1][i][1]-1))
    fruit_snake_score_human[2].append(arg[2])
    game.points_fruit_human = fruit_snake_score_human[0]
    game.points_snake_human = fruit_snake_score_human[1]
    game.score_human = fruit_snake_score_human[2][0][0]
    game.draw_elements_human()
    game._display()


@sio.event
def get_env_bot(arg):
    fruit_snake_score_bot = [[], [], [], []]
    fruit_snake_score_bot[0].append(Point(arg[0][0][0]-1, arg[0][0][1]-1))
    for i in range(len(arg[1])):
        fruit_snake_score_bot[1].append(Point(arg[1][i][0]-1, arg[1][i][1]-1))
    fruit_snake_score_bot[2].append(arg[2])
    fruit_snake_score_bot[3].append(arg[3])
    game.points_fruit_bot = fruit_snake_score_bot[0]
    game.points_snake_bot = fruit_snake_score_bot[1]
    game.score_bot = fruit_snake_score_bot[2][0][0]
    if fruit_snake_score_bot[3][0][0][0] == 1 and fruit_snake_score_bot[3][0][0][1] == 0:
        game.action_bot = 'Right'
    elif fruit_snake_score_bot[3][0][0][0] == -1 and fruit_snake_score_bot[3][0][0][1] == 0:
        game.action_bot = 'Left'
    elif fruit_snake_score_bot[3][0][0][0] == 0 and fruit_snake_score_bot[3][0][0][1] == 1:
        game.action_bot = 'Down'
    elif fruit_snake_score_bot[3][0][0][0]== 0 and fruit_snake_score_bot[3][0][0][1] == -1:
        game.action_bot = 'Up'
    game.draw_elements_bot()
    game._display()

@sio.event
def hum_win():
    game.pause_human_win()
    game._display()

@sio.event
def hum_lose():
    game.pause_human_lose()
    game._display()

@sio.event
def bot_win():
    game.pause_bot_win()
    game._display()

@sio.event
def bot_lose():
    game.pause_bot_lose()
    game._display()

@sio.event
def tray_again():
    game.end()
    game._display()


@sio.event
def connect():
    print("I'm connected!")

@sio.event
def connect_error(data):
    print("The connection failed!")

@sio.event
def disconnect():
    print("I'm disconnected!")

sio.connect('http://127.0.0.1:5000', wait=True, wait_timeout= 5)

def menu(menu, pos='center', font1=None, font2=None, color1=(0, 0, 0), color2=None, interline=5, justify=True, light=5, speed=300, lag=30):

    class Item(Rect):

        def __init__(self, menu, label):
            Rect.__init__(self, menu)
            self.label = label

    def show():
        i = Rect((0, 0), font2.size(menu[idx].label))
        if justify:
            i.center = menu[idx].center
        else:
            i.midleft = menu[idx].midleft
        pygame.display.update(
            (scr.blit(bg, menu[idx], menu[idx]), scr.blit(font2.render(menu[idx].label, 1, (0, 0, 0)), i)))

        #time.wait(50)
        scr.blit(bg, r2, r2)
        [scr.blit(font1.render(item.label, 1, color1), item)
         for item in menu if item != menu[idx]]
        r = scr.blit(font2.render(menu[idx].label, 1, color2), i)
        pygame.display.update(r2)

        return r

    def anim():
        clk = pygame.time.Clock()
        a = [menu[0]] if lag else menu[:]
        c = 0
        while a:
            for i in a:
                g = i.copy()
                i.x = i.animx.pop(0)
                r = scr.blit(font1.render(i.label, 1, color1), i)
                pygame.display.update((g, r))

                scr.blit(bg, r, r)
            c += 1
            if not a[0].animx:
                a.pop(0)
                if not lag:
                    break
            if lag:
                foo, bar = divmod(c, lag)
                if not bar and foo < len(menu):
                    a.append(menu[foo])
            clk.tick(speed)

    events = pygame.event.get()
    scr = pygame.display.get_surface()
    scrrect = scr.get_rect()
    bg = scr.copy()
    if not font1:
        font1 = pygame.font.Font(None, scrrect.h // len(menu) // 3)
    if not font2:
        font2 = font1
    if not color1:
        color1 = (0, 0, 0)
    if not color2:
        color2 = (0,0,0)#list(map(lambda x: x + (255 - x) * light // 10, color1))
    m = max(menu, key=font1.size)
    r1 = Rect((0, 0), font1.size(m))
    ih = r1.size[1]
    r2 = Rect((0, 0), font2.size(m))
    r2.union_ip(r1)
    w, h = r2.w - r1.w, r2.h - r1.h
    r1.h = (r1.h + interline) * len(menu) - interline
    r2 = r1.inflate(w, h)

    try:
        setattr(r2, pos, getattr(scrrect, pos))
    except:
        r2.topleft = pos
    if justify:
        r1.center = r2.center
    else:
        r1.midleft = r2.midleft

    menu = [Item(((r1.x, r1.y + e * (ih + interline)), font1.size(i)), i)
            for e, i in enumerate(menu)if i]
    if justify:
        for i in menu:
            i.centerx = r1.centerx

    if speed:
        for i in menu:
            z = r1.w - i.x + r1.x
            i.animx = [
                cos(radians(x)) * (i.x + z) - z for x in list(range(90, -1, -1))]
            i.x = i.animx.pop(0)
        anim()
        for i in menu:
            z = scrrect.w + i.x - r1.x
            i.animx = [
                cos(radians(x)) * (-z + i.x) + z for x in list(range(0, -91, -1))]
            i.x = i.animx.pop(0)

    mpos = Rect(pygame.mouse.get_pos(), (0, 0))
    pygame.event.post(
        pygame.event.Event(MOUSEMOTION, {'pos': mpos.topleft if mpos.collidelistall(menu) else menu[0].center}))
    idx = -1

    while True:
        ev = pygame.event.wait()
        if ev.type == MOUSEMOTION:
            idx_ = Rect(ev.pos, (0, 0)).collidelist(menu)
            if idx_ > -1 and idx_ != idx:
                idx = idx_
                r = show()
        elif ev.type == MOUSEBUTTONUP and r.collidepoint(ev.pos):
            ret = menu[idx].label, idx
            break
        elif ev.type == KEYDOWN:
            try:
                idx = (idx + {K_UP: -1, K_DOWN: 1}[ev.key]) % len(menu)
                r = show()
            except:
                if ev.key in (K_RETURN, K_KP_ENTER):
                    ret = menu[idx].label, idx
                    break
                elif ev.key == K_ESCAPE:
                    ret = None, None
                    break
    scr.blit(bg, r2, r2)

    if speed:
        [scr.blit(font1.render(i.label, 1, color1), i) for i in menu]
        pygame.display.update(r2)
        pygame.time.wait(50)
        scr.blit(bg, r2, r2)
        anim()
    else:
        pygame.display.update(r2)

    for ev in events:
        pygame.event.post(ev)
    return ret


class run(object):

    def runm(self):
        pygame.init()
        scr = pygame.display.set_mode((1800, 1010))
        pygame.display.set_caption("NeuroFreak")
        icon = pygame.image.load("src/Assets/snake_icon.png")
        pygame.display.set_icon(icon)
        f = pygame.font.Font(join("src/Font/magic.TTF"), 125)
        f1 = pygame.font.Font(join("src/Font/magic.TTF"), 65)
        mainmenu = f.render('NeuroFreak', 1, (255, 255, 255))
        r = mainmenu.get_rect()
        r.centerx, r.top = 900, 80
        background_main = pygame.image.load("src/Background/bg.jpg").convert()
        background_main = pygame.transform.scale(background_main, (1800,1010))  
        scr.blit(background_main, (0, 0))
        bg = scr.copy()
        scr.blit(mainmenu, r)
        pygame.display.flip()

        size = ['Размер поля: 10x10', 'Размер поля: 15x15', 'Размер поля: 20x20']
        speed = ['Скорость игры: 1x', 'Скорость игры: 2x', 'Скорость игры: 3x']
        bot = ['Тип игрового бота: Гамильноновы циклы', 'Тип игрового бота: Поиск по дереву Монте Карло', 'Тип игрового бота: Глубокое Q-обучение']
        cur_size = size[0]
        cur_speed = speed[0]
        cur_bot = bot[0]

        is_it_menu = True

        menu1 = {"menu": ['Играть', 'Настройки', 'Выход'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
        menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
        resp = "re-show"
        global cell, fps, bot_name
        cell = 10
        pixel_size = 40
        fps = 6
        bot_name = 'Hamilton'
        while is_it_menu:
            if resp == "re-show":
                resp = menu(**menu1)[0]

            if resp == 'Настройки':
                resp = menu(**menu2)[0]

            if resp == 'Вернуться':
                resp = menu(**menu1)[0]

            if resp == size[0]:

                cell = 15
                cur_size = size[1]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]
            
            if resp == size[1]:
                cell = 20
                cur_size = size[2]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == size[2]:
                cell = 10
                cur_size = size[0]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == speed[0]:
                fps = 9
                cur_speed = speed[1]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == speed[1]:
                fps = 12
                cur_speed = speed[2]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == speed[2]:
                fps = 6
                cur_speed = speed[0]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == bot[0]:
                bot_name = 'Monte Carlo'
                cur_bot = bot[1]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == bot[1]:
                bot_name = 'DQN'
                cur_bot = bot[2]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == bot[2]:
                bot_name = 'Hamilton'
                cur_bot = bot[0]
                menu2 = {"menu": [cur_size, cur_speed, cur_bot, 'Вернуться'], "font1": f1, "pos":
                 'center', "color1": (200, 200, 200), "light": 6, "speed": 200, "lag": 20}
                resp = menu(**menu2)[0]

            if resp == 'Выход':
                pygame.display.quit()
                pygame.quit()
                sio.disconnect()
                sys.exit()

            if resp == 'Играть':
                global game
                is_it_menu = False
                game = Game(pixel_size, cell, bot_name)
                game.draw_elements_human()
                game.draw_elements_bot()
                game._display()
                pygame.event.clear()
                eve = pygame.event.wait()
                if eve.type == QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sio.disconnect()
                    sys.exit()
                elif eve.type == KEYDOWN:
                    if eve.key == K_ESCAPE:
                        mymenu.runm()
                    if eve.key == K_RIGHT:
                        arg = [cell, bot_name]
                        sio.emit("start", arg)
                        play()
                        #action = [0, 0]
                        #sio.emit("game", action)

if __name__ == "__main__":

    mymenu = run()
    mymenu.runm()
