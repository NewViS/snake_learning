from environment import Environment
import random

env = Environment()
env.set_wall()
for i in range(100):
    env.set_fruit()
    print(env.fruit[0].x, env.fruit[0].y)

# a= [[x, y] for x in range(1, 11) for y in range(1,11)]
# for i in range(100):
    # elem = random.choice(a)
    # tmp = a.index(elem)
    # a[tmp]=a[-1]
    # a[-1]=elem
# print(a)