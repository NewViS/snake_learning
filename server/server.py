from action import Action
from environment import Environment
from flask import Flask
from flask_socketio import SocketIO, emit
from minimax import MiniMax
from human import Human
from hamilton import Hamilton
from DQN_10 import DQN_play

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@socketio.on('connect')
def connect(auth):
    print('I connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on("start")
def starting(arg):
    global environment_human, environment_bot, human, model
    environment_human = Environment(width=arg[0]+2, height=arg[0]+2)
    environment_bot = Environment(width=arg[0]+2, height=arg[0]+2)
    human = Human()
    if arg[1] == 'Hamilton':
        model = Hamilton()
    elif arg[1] == 'Monte Carlo':
        model = MiniMax()
    elif arg[1] == 'DQN':
        model = DQN_play()

@socketio.on("game")
def playing(action):
    if action[0] == 0 and action[1] == 0:
            pass
    elif action[0] == 1 and action[1] == 0:
        human.action = Action.right
    elif action[0] == -1 and action[1] == 0:
        human.action = Action.left
    elif action[0] == 0 and action[1] == 1:
        human.action = Action.down
    elif action[0] == 0 and action[1] == -1:
        human.action = Action.up
    if not environment_human.Terminal:
        ai_action = human.move(environment_human)
        environment_human.full_step(ai_action)
        if environment_human.Terminal:
            if environment_human._is_winning():
                emit("hum_win")
            else:
                emit("hum_lose")
    if not environment_bot.Terminal:
        ai_action = model.move(environment_bot)
        environment_bot.full_step(ai_action)
        if environment_bot.Terminal:
            if environment_bot._is_winning():
                emit("bot_win")
            else:
                emit("bot_lose")
    if not environment_human.Terminal and not environment_bot.Terminal:
        state = [[[], [], []],[[], [], [],[]]]
        state[0][0].append((environment_human.fruit[0][0], environment_human.fruit[0][1]))
        for i in range(0, environment_human.snake_length):
            state[0][1].append((environment_human.snake[i][0], environment_human.snake[i][1]))
        state[0][2].append(environment_human.snake_length)
        state[1][0].append((environment_bot.fruit[0][0], environment_bot.fruit[0][1]))
        for i in range(0, environment_bot.snake_length):
            state[1][1].append((environment_bot.snake[i][0], environment_bot.snake[i][1]))
        state[1][2].append(environment_bot.snake_length)
        if environment_bot.snake_action == Action.up:
            act = [0,-1]
        elif environment_bot.snake_action == Action.down:
            act = [0,1]
        elif environment_bot.snake_action == Action.right:
            act = [1,0]
        elif environment_bot.snake_action == Action.left:
            act=[-1,0]
        state[1][3].append(act)
        emit("get_both_env", state)
    elif not environment_human.Terminal and environment_bot.Terminal:
        state = [[], [], []]
        state[0].append((environment_human.fruit[0][0], environment_human.fruit[0][1]))
        for i in range(0, environment_human.snake_length):
            state[1].append((environment_human.snake[i][0], environment_human.snake[i][1]))
        state[2].append(environment_human.snake_length)
        emit("get_env_human", state)
    elif environment_human.Terminal and not environment_bot.Terminal:
        state = [[], [], [], []]
        state[0].append((environment_bot.fruit[0][0], environment_bot.fruit[0][1]))
        for i in range(0, environment_bot.snake_length):
            state[1].append((environment_bot.snake[i][0], environment_bot.snake[i][1]))
        state[2].append(environment_bot.snake_length)
        if environment_bot.snake_action == Action.up:
            act = [0,-1]
        elif environment_bot.snake_action == Action.down:
            act = [0,1]
        elif environment_bot.snake_action == Action.right:
            act = [1,0]
        elif environment_bot.snake_action == Action.left:
            act=[-1,0]
        state[3].append(act)
        emit("get_env_bot", state)
    elif environment_human.Terminal and environment_bot.Terminal:
        emit("try_again")


if __name__ == '__main__':
    socketio.run(app)