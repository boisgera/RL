import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import itertools

    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F

    return nn, plt, torch


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breakout game, stage 0

    There is a grid of width 3 and height 2. At time 0, a ball is dropped at the top of the grid, with a random uniform location. The ball will drop at the bottom at time 1. The paddle is initially located at the center of the bottom of the grid and may move one cell to the left, stay still or move one cell to the right to try to catch the ball.

    Obviously the approppriate strategy to catch the ball here is: move to the left if the ball is on the left, stay still if the ball is in the center, and move to the right if the ball is one the right.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can encode manually a neural network with

    - 1 input : the location of the ball, between -1.0 (left) and 1.0 (right)
    - 3 outputs : the logit of each possible action (move left, stay still, move right)
    - no hidden layer

    that does this job arbitrarily well. The trick is to find three affine functions of the target location such that in each region of interest, a given function is far greater than the others. For example:
    """)
    return


@app.cell
def _(torch):
    DELTA = 10.0  # increase DELTA to get "sharper" decisions.

    def move_left_logit(x_target, delta=DELTA):
        return - delta * (2 * x_target + 1)

    def stay_still_logit(x_target, delta=DELTA):
        return torch.zeros_like(x_target)

    def move_right_logit(x_target, delta=DELTA):
        return delta * (2 * x_target - 1)


    return DELTA, move_left_logit, move_right_logit, stay_still_logit


@app.cell
def _(mo, move_left_logit, move_right_logit, plt, stay_still_logit, torch):
    x_target = torch.linspace(-1.0, 1.0, 1000)
    plt.plot(x_target, move_left_logit(x_target), label="move_left logit")
    plt.plot(x_target, stay_still_logit(x_target), label="stay_still logit")
    plt.plot(x_target, move_right_logit(x_target), label="move_right logit")
    plt.grid(True)
    plt.legend()
    mo.center(plt.gcf())
    return (x_target,)


@app.cell
def _(
    mo,
    move_left_logit,
    move_right_logit,
    plt,
    stay_still_logit,
    torch,
    x_target,
):
    def _():
        left_logit = move_left_logit(x_target)
        center_logit = stay_still_logit(x_target)
        right_logit = move_right_logit(x_target)
        logits = torch.stack((left_logit, center_logit, right_logit), dim=1)
        probas = torch.softmax(logits, dim=1)
        probas = torch.softmax(logits, dim=1)
        plt.title("Probability of action")
        plt.xlabel("target position")
        plt.plot(x_target, probas[:, 0], label="move left")
        plt.plot(x_target, probas[:, 1], label="stay still")
        plt.plot(x_target, probas[:, 2], label="move right")
        plt.legend()
        return mo.center(plt.gcf())
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's encode this in a Pytorch model.
    """)
    return


@app.cell
def _(DELTA, nn, torch):
    def DecisionModel():
        return nn.Linear(in_features=1, out_features=3)

    def setup(model, delta=DELTA):
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[-2*delta], [0.0], [2*delta]])) # slopes
            model.bias.copy_(torch.tensor([-delta, 0.0, -delta])) # offsets

    action = DecisionModel()
    setup(action)
    return (action,)


@app.cell
def _(action, mo, plt, torch):
    def _():
        n = 1000
        x_target = torch.linspace(-1.0, 1.0, n).reshape((n, 1))
        with torch.inference_mode():
            logits = action(x_target)
        probas = torch.softmax(logits, dim=1)
        plt.title("Probability of action")
        plt.xlabel("target position")
        plt.plot(x_target, probas[:, 0], label="move left")
        plt.plot(x_target, probas[:, 1], label="stay still")
        plt.plot(x_target, probas[:, 2], label="move right")
        plt.legend()
        return mo.center(plt.gcf())
    _()
    return


if __name__ == "__main__":
    app.run()
