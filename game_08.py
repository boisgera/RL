# Python Standard Library
import random

# Third-Party
import pyxel
import torch


# Constants
# ------------------------------------------------------------------------------
PIXEL_SIZE = 40
INFO_HEIGHT = 16  # space above the arena for two lines of text
GRID_SHAPE = (5, 5)
FPS = 30
STEP_FRAMES = 30  # frames per time step
STATUS_COLOR = {True: pyxel.COLOR_LIME, False: pyxel.COLOR_RED}


# Policy Model
# ------------------------------------------------------------------------------
policy = torch.nn.Sequential(
    torch.nn.Linear(in_features=3, out_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=32, out_features=32),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=32, out_features=3),
)
policy.load_state_dict(torch.load("models/model08.pt"))


# Game State
# ------------------------------------------------------------------------------
manual = False  # user 🧑 or model 🤖 controlled game
ball_i = None  # ball column index
ball_di = None  # ball horizontal velocity in {-1, 0, 1}
paddle_i = None  # paddle column index
t = None  # time step (ball row index)
move = None  # next paddle move
frame0 = None  # frame count at the start of the time step


# Main Functions
# ------------------------------------------------------------------------------
def to_x(i):
    "Map the column index i to x in [-1.0, 1.0] (leftmost/rightmost)"
    return 2.0 * i / (GRID_SHAPE[0] - 1) - 1.0


def next_move():
    if manual:  # 🧑
        return 0  # by default, don't move
    else:  # 🤖
        with torch.no_grad():
            input = torch.tensor([to_x(paddle_i), to_x(ball_i), float(ball_di)])
            choice = policy(input).argmax()
            return int(choice) - 1


def move_ball():
    "Move the ball horizontally, with a rebound on the walls"
    global ball_i, ball_di
    ball_i += ball_di
    if ball_i > GRID_SHAPE[0] - 1:
        ball_i = 2 * (GRID_SHAPE[0] - 1) - ball_i
        ball_di = -ball_di
    elif ball_i < 0:
        ball_i = -ball_i
        ball_di = -ball_di


def new_round():
    global frame0, ball_i, ball_di, paddle_i, t, move
    frame0 = pyxel.frame_count
    ball_i = random.randrange(GRID_SHAPE[0])
    ball_di = random.choice([-1, 0, 1])
    paddle_i = random.randrange(GRID_SHAPE[0])
    t = 0
    move = next_move()


def update():
    global manual, frame0, paddle_i, t, move
    if pyxel.btnp(pyxel.KEY_SPACE):
        manual = not manual
    if manual:
        if pyxel.btnp(pyxel.KEY_LEFT):
            move = -1
        if pyxel.btnp(pyxel.KEY_RIGHT):
            move = 1
    if pyxel.frame_count - frame0 >= STEP_FRAMES:
        if t == GRID_SHAPE[1] - 1:
            new_round()
        else:
            frame0 = pyxel.frame_count
            paddle_i = min(max(paddle_i + move, 0), GRID_SHAPE[0] - 1)
            move_ball()
            t += 1
            move = next_move()


def draw_header(over, success):
    mode = "MANUAL" if manual else "AUTO"
    pyxel.text(2, 2, f"{mode} (hit space to toggle)", pyxel.COLOR_WHITE)
    if over:
        status = "SUCCESS" if success else "FAILURE"
        pyxel.text(2, 9, status, STATUS_COLOR[success])
    else:
        action = {-1: "<-", 0: "--", 1: "->"}[move]
        pyxel.text(2, 9, action, pyxel.COLOR_WHITE)


def draw():
    over = t == GRID_SHAPE[1] - 1
    success = paddle_i == ball_i
    background_color = STATUS_COLOR[success] if over else pyxel.COLOR_BLACK
    pyxel.cls(pyxel.COLOR_BLACK)
    draw_header(over, success)
    pyxel.rect(
        0,
        INFO_HEIGHT,
        GRID_SHAPE[0] * PIXEL_SIZE,
        GRID_SHAPE[1] * PIXEL_SIZE,
        background_color,
    )
    pyxel.rect(
        paddle_i * PIXEL_SIZE,
        INFO_HEIGHT + (GRID_SHAPE[1] - 1) * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_WHITE,
    )
    pyxel.rect(
        ball_i * PIXEL_SIZE,
        INFO_HEIGHT + t * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_YELLOW,
    )


pyxel.init(
    GRID_SHAPE[0] * PIXEL_SIZE,
    INFO_HEIGHT + GRID_SHAPE[1] * PIXEL_SIZE,
    title="Breakout 08",
    fps=FPS,
)
new_round()
pyxel.run(update, draw)
