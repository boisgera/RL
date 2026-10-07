import random

import pyxel
import torch

CELL = 5     # pixel size of a grid cell
FRAMES = 1  # frames per time step (time 0: ball at the top, time 1: ball at the bottom)

policy = torch.nn.Linear(in_features=1, out_features=3)
policy.load_state_dict(torch.load("models/model00.pt"))

manual = False


def new_round():
    global frame0, ball_x, move
    frame0 = pyxel.frame_count
    ball_x = random.randint(-1, 1)
    with torch.no_grad():
        move = policy(torch.tensor([[float(ball_x)]])).argmax().item() - 1
    if manual:
        move = 0


def update():
    global manual, move
    if pyxel.btnp(pyxel.KEY_SPACE):
        manual = not manual
    t = (pyxel.frame_count - frame0) // FRAMES
    if manual and t == 0:
        if pyxel.btnp(pyxel.KEY_LEFT):
            move = -1
        if pyxel.btnp(pyxel.KEY_RIGHT):
            move = 1
    if t == 2:
        new_round()


def draw():
    t = (pyxel.frame_count - frame0) // FRAMES
    paddle_x = move if t == 1 else 0
    pyxel.cls(0 if t == 0 else 11 if paddle_x == ball_x else 8)
    pyxel.rect((paddle_x + 1) * CELL, CELL, CELL, CELL, 7)
    pyxel.rect((ball_x + 1) * CELL, t * CELL, CELL, CELL, 10)


pyxel.init(3 * CELL, 2 * CELL, title="Breakout 00", display_scale=40)
new_round()
pyxel.run(update, draw)
