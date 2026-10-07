# Python Standard Library


# Third-Party
import pyxel
import torch


# Constants
# ------------------------------------------------------------------------------
PIXEL_SIZE = 5
GRID_SHAPE = (3, 2)
FPS = 1

# Policy Model
# ------------------------------------------------------------------------------
policy = torch.nn.Linear(in_features=1, out_features=3)
policy.load_state_dict(torch.load("models/model00.pt"))


# Game State
# ------------------------------------------------------------------------------
manual = False # game controlled by the user (keyboard) or the models
ball_x = None
move = None
frame0 = None


# Main Functions
# ------------------------------------------------------------------------------
def new_round():
    global frame0, ball_x, move
    frame0 = pyxel.frame_count
    ball_x = torch.randint(-1, 1, ())
    if manual:
        move = 0  # by default, don't move
    else:  # auto
        with torch.no_grad():
            input = torch.tensor([ball_x]).float()
            choice = policy(input).argmax()
            move = choice - 1.0


def update():
    global manual, move
    if pyxel.btnp(pyxel.KEY_SPACE):
        manual = not manual
    t = pyxel.frame_count - frame0
    if manual and t == 0:
        if pyxel.btnp(pyxel.KEY_LEFT):
            move = -1
        if pyxel.btnp(pyxel.KEY_RIGHT):
            move = 1
    if t == 2:
        new_round()


def draw():
    t = pyxel.frame_count - frame0
    paddle_x = move if t == 1 else 0
    pyxel.cls(pyxel.COLOR_BLACK if t == 0 else pyxel.COLOR_LIME if paddle_x == ball_x else pyxel.COLOR_RED)
    pyxel.rect((paddle_x + 1) * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, pyxel.COLOR_WHITE)
    pyxel.rect((ball_x + 1) * PIXEL_SIZE, t * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, pyxel.COLOR_YELLOW)


pyxel.init(
    GRID_SHAPE[0] * PIXEL_SIZE,
    GRID_SHAPE[1] * PIXEL_SIZE,
    title="Breakout 00",
    # display_scale=40,
    fps=FPS,
)
new_round()
pyxel.run(update, draw)
