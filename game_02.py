# Python Standard Library
import random

# Third-Party
import pyxel
import torch


# Constants
# ------------------------------------------------------------------------------
PIXEL_SIZE = 40
INFO_HEIGHT = 16  # space above the arena for two lines of text
GRID_SHAPE = (5, 2)
FPS = 30
STEP_FRAMES = 30  # frames per time step
STATUS_COLOR = {True: pyxel.COLOR_LIME, False: pyxel.COLOR_RED}


# Policy Model
# ------------------------------------------------------------------------------
policy = torch.nn.Linear(in_features=2, out_features=3)
policy.load_state_dict(torch.load("models/model02.pt"))


# Game State
# ------------------------------------------------------------------------------
manual = False  # user 🧑 or model 🤖 controlled game
ball_i = None  # ball column index
paddle_i = None  # initial paddle column index
move = None
frame0 = None


# Main Functions
# ------------------------------------------------------------------------------
def to_x(i):
    "Map the column index i to x in [-1.0, 1.0] (leftmost/rightmost)"
    return 2.0 * i / (GRID_SHAPE[0] - 1) - 1.0


def new_round():
    global frame0, ball_i, paddle_i, move
    frame0 = pyxel.frame_count
    ball_i = random.randrange(GRID_SHAPE[0])
    paddle_i = random.randrange(GRID_SHAPE[0])
    if manual:  # 🧑
        move = 0  # by default, don't move
    else:  # 🤖
        with torch.no_grad():
            input = torch.tensor([to_x(paddle_i), to_x(ball_i)])
            choice = policy(input).argmax()
            move = int(choice) - 1


def update():
    global manual, move
    if pyxel.btnp(pyxel.KEY_SPACE):
        manual = not manual
    elapsed_time = (pyxel.frame_count - frame0) // STEP_FRAMES
    if elapsed_time == 2:
        new_round()
        elapsed_time = 0
    if manual and elapsed_time == 0:
        if pyxel.btnp(pyxel.KEY_LEFT):
            move = -1
        if pyxel.btnp(pyxel.KEY_RIGHT):
            move = 1


def draw_header(elapsed_time, success):
    mode = "MANUAL" if manual else "AUTO"
    pyxel.text(2, 2, f"{mode} (hit space to toggle)", pyxel.COLOR_WHITE)
    if elapsed_time == 1:
        action = {-1: "<-", 0: "--", 1: "->"}[move]
        pyxel.text(2, 9, action, pyxel.COLOR_WHITE)
        status = "SUCCESS" if success else "FAILURE"
        pyxel.text(14, 9, status, STATUS_COLOR[success])


def draw():
    elapsed_time = (pyxel.frame_count - frame0) // STEP_FRAMES
    paddle_column = paddle_i
    if elapsed_time == 1:
        paddle_column = min(max(paddle_i + move, 0), GRID_SHAPE[0] - 1)
    success = paddle_column == ball_i
    background_color = pyxel.COLOR_BLACK if elapsed_time == 0 else STATUS_COLOR[success]
    pyxel.cls(pyxel.COLOR_BLACK)
    draw_header(elapsed_time, success)
    pyxel.rect(
        0,
        INFO_HEIGHT,
        GRID_SHAPE[0] * PIXEL_SIZE,
        GRID_SHAPE[1] * PIXEL_SIZE,
        background_color,
    )
    pyxel.rect(
        paddle_column * PIXEL_SIZE,
        INFO_HEIGHT + (GRID_SHAPE[1] - 1) * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_WHITE,
    )
    pyxel.rect(
        ball_i * PIXEL_SIZE,
        INFO_HEIGHT + elapsed_time * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_YELLOW,
    )


pyxel.init(
    GRID_SHAPE[0] * PIXEL_SIZE,
    INFO_HEIGHT + GRID_SHAPE[1] * PIXEL_SIZE,
    title="Breakout 02",
    fps=FPS,
)
new_round()
pyxel.run(update, draw)
