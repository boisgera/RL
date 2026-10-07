# Python Standard Library


# Third-Party
import pyxel
import torch


# Constants
# ------------------------------------------------------------------------------
PIXEL_SIZE = 40
INFO_HEIGHT = 16  # space above the arena for two lines of text
GRID_SHAPE = (3, 2)
FPS = 30
STEP_FRAMES = 30  # frames per time step
STATUS_COLOR = {True: pyxel.COLOR_LIME, False: pyxel.COLOR_RED}


# Policy Model
# ------------------------------------------------------------------------------
policy = torch.nn.Linear(in_features=1, out_features=3)
policy.load_state_dict(torch.load("models/model00.pt"))


# Game State
# ------------------------------------------------------------------------------
manual = False  # user 🧑 or model 🤖 controlled game
ball_x = None
move = None
frame0 = None


# Main Functions
# ------------------------------------------------------------------------------
def new_round():
    global frame0, ball_x, move
    frame0 = pyxel.frame_count
    ball_x = torch.randint(-1, 2, ())
    if manual:  # 🧑
        move = 0  # by default, don't move
    else:  # 🤖
        with torch.no_grad():
            input = torch.tensor([ball_x]).float()
            choice = policy(input).argmax()
            move = choice - 1.0


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
            move = -1.0
        if pyxel.btnp(pyxel.KEY_RIGHT):
            move = 1.0


def column(x):
    "Map x from [-1.0, 1.0] (leftmost/rightmost) to the cell column index"
    index = round((float(x) + 1.0) / 2.0 * (GRID_SHAPE[0] - 1))
    return index


def draw():
    elapsed_time = (pyxel.frame_count - frame0) // STEP_FRAMES
    paddle_x = move if elapsed_time == 1 else 0
    success = (paddle_x == ball_x).item()
    background_color = pyxel.COLOR_BLACK if elapsed_time == 0 else STATUS_COLOR[success]
    pyxel.cls(pyxel.COLOR_BLACK)
    mode = "MANUAL" if manual else "AUTO"
    pyxel.text(2, 2, f"{mode} (hit space to toggle)", pyxel.COLOR_WHITE)
    if elapsed_time == 1:
        action = {-1: "<-", 0: "--", 1: "->"}[int(move)]
        pyxel.text(2, 9, action, pyxel.COLOR_WHITE)
        status = "SUCCESS" if success else "FAILURE"
        pyxel.text(14, 9, status, background_color)
    pyxel.rect(
        0,
        INFO_HEIGHT,
        GRID_SHAPE[0] * PIXEL_SIZE,
        GRID_SHAPE[1] * PIXEL_SIZE,
        background_color,
    )
    pyxel.rect(
        column(paddle_x) * PIXEL_SIZE,
        INFO_HEIGHT + (GRID_SHAPE[1] - 1) * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_WHITE,
    )
    pyxel.rect(
        column(ball_x) * PIXEL_SIZE,
        INFO_HEIGHT + elapsed_time * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE,
        pyxel.COLOR_YELLOW,
    )


pyxel.init(
    GRID_SHAPE[0] * PIXEL_SIZE,
    INFO_HEIGHT + GRID_SHAPE[1] * PIXEL_SIZE,
    title="Breakout 00",
    # display_scale=40,
    fps=FPS,
)
new_round()
pyxel.run(update, draw)
