import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    return pd, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breakout game, stage 0

    The game area is a grid of width 3 and height 2. At time 0, a ball is dropped at the top of the grid, with a random uniform location
    (left, center or middle). The ball will drop vertically to the bottom at time 1. The paddle, which occupies a single cell, is initially located at the center of the bottom of the grid. It may use the ball location at time 0 to decide wheter wants to move one cell to the left, stay still or move one cell to the right to try to catch the ball.

    Obviously the appropriate strategy to catch the ball here is:
    - move to the left if the ball is on the left,
    - stay still if the ball is in the center, and
    - move to the right if the ball is one the right.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We want to develop a neural network that will decide what action the paddle should take given the location of the target.
    This will be a classic decision model, which output logits (unnormalized log-probabilities) for each possible action.

    - 1 input : the location of the ball, between -1.0 (left) and 1.0 (right)
    - 3 outputs : the logit of every possible action (move left, stay still, move right)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's actually possible to create manually a neural network with no hidden layer that does the job arbitrarily well. The trick is to find three affine functions of the target location such that in each region of interest, the value of the appropriare function is far greater than the others. For example:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    log10_delta_slider = mo.ui.slider(start=0, stop=10, step=1, value=1, show_value=True, label=r"$\log_{10} \Delta$")
    mo.md(rf'''
    Increase $\Delta$ to get "sharper" decisions 

    {log10_delta_slider}''')
    return (log10_delta_slider,)


@app.cell
def _(log10_delta_slider, torch):
    delta = 10**log10_delta_slider.value

    def action_logits(delta=delta):
        def move_left_logit(x_target):
            return - delta * (2 * x_target + 1)
    
        def stay_still_logit(x_target):
            return torch.zeros_like(x_target)
    
        def move_right_logit(x_target):
            return delta * (2 * x_target - 1)
        return move_left_logit, stay_still_logit, move_right_logit

    move_left_logit, stay_still_logit, move_right_logit = action_logits()

    return delta, move_left_logit, move_right_logit, stay_still_logit


@app.cell
def _(mo, move_left_logit, move_right_logit, plt, stay_still_logit, torch):
    def _():
        x_target = torch.linspace(-1.0, 1.0, 1000)
        plt.plot(x_target, move_left_logit(x_target), label="move_left logit")
        plt.plot(x_target, stay_still_logit(x_target), label="stay_still logit")
        plt.plot(x_target, move_right_logit(x_target), label="move_right logit")
        plt.grid(True)
        plt.legend()
        return mo.center(plt.gcf())
    _()
    return


@app.cell
def _(mo, move_left_logit, move_right_logit, plt, stay_still_logit, torch):
    def _():
        x_target = torch.linspace(-1.0, 1.0, 1000)
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
        plt.grid(True)
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
def _(delta, torch):
    def DecisionModel():
        return torch.nn.Linear(in_features=1, out_features=3)

    def setup(model, delta=delta):
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[-2*delta], [0.0], [2*delta]])) # slopes
            model.bias.copy_(torch.tensor([-delta, 0.0, -delta])) # offsets


    return DecisionModel, setup


@app.cell
def _(DecisionModel, mo, plt, setup, torch):
    def _():
        policy = DecisionModel()
        setup(policy)
        torch.save(policy.state_dict(), "models/model00.pt")
    
        n = 1000
        x_target = torch.linspace(-1.0, 1.0, n).reshape((n, 1))
        with torch.inference_mode():
            logits = policy(x_target)
        probas = torch.softmax(logits, dim=1)
        plt.title("Probability of action")
        plt.xlabel("target position")
        plt.plot(x_target, probas[:, 0], label="move left")
        plt.plot(x_target, probas[:, 1], label="stay still")
        plt.plot(x_target, probas[:, 2], label="move right")
        plt.legend()
        plt.grid(True)
        return mo.center(plt.gcf())
    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell
def _(DecisionModel, pd, torch):
    def _():
        policy = DecisionModel()
        state_dict = torch.load("models/model00.pt")
        policy.load_state_dict(state_dict)

        num_samples = 100 # batch size
        target_x = torch.randint(-1, 2, (num_samples, 1)).float()
        action_logits = policy(target_x)
        choice = action_logits.argmax(dim=1) # deterministic choice
        paddle_x = choice.float() - 1.0
        target_x = target_x.squeeze()
        df = pd.DataFrame({"target_x": target_x, "paddle_x": paddle_x, "success": paddle_x == target_x})
        return df

    _()
    return


if __name__ == "__main__":
    app.run()
