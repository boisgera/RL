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
    import torch.nn.functional as F

    return F, pd, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Breakout game, stage 7

    The game now takes place on a grid of width `WIDTH` and height `HEIGHT` (larger than 2).
    The paddle starts at a random (uniform) position on the lower level and the ball pops at a random (uniform)
    position on the upper level, independently. The ball may drop vertically, one cell at time, or drop to the left or to the right with a 45 degrees angle ; it will rebound on a wall. It reaches the lower level after `HEIGHT - 1` ticks; at every tick, the paddle can decide to stay at its position, move left or move right (within the grid). To make this decision it may use info about the ball position and horizontal velocity. Therefore the paddle may move `HEIGHT - 1` times to try and catch the ball.

    The paddle and target positions are integers in $\{0, \dots, {\rm WIDTH}-1\}$, always mapped linearly to $[-1.0, 1.0]$. The ball velocity is either $-1.0$, $0.0$ or $1.0$.
    """)
    return


@app.cell
def _():
    WIDTH = 5
    HEIGHT = 5
    assert HEIGHT >= WIDTH  # Simpler: we know that we can always catch the ball.
    return HEIGHT, WIDTH


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The decision model is now larger and deeper: two hidden layers of 32 units each (with ReLU activation),
    used at every tick:

    - 3 inputs : the current paddle position $x_t$ and target position $\texttt{target\_x}$, between -1.0 (left) and 1.0 (right), and the target velocity $\texttt{target\_dx}$ in $\{-1.0, 0.0, 1.0\}$
    - 3 outputs : the logit of every possible action (move left, stay still, move right)

    In stage 4, the linear model could not reach the maximal mean reward: because of the rebounds on the walls,
    the position where the ball lands is not a linear function of its current position and velocity.
    In stage 5, we have shown that 3 hidden units are enough to apply an optimal strategy, but in stage 6
    the training of this architecture got stuck on a plateau (mean reward around 0.62 to 0.68).
    We try to escape this plateau with a larger network.
    """)
    return


@app.cell
def _(torch):
    def DecisionModel():
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=3, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=3),
        )

    return (DecisionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reward

    The reward is 1 if the paddle catches the ball (at the last tick) and 0 otherwise.

    Since `HEIGHT >= WIDTH`, the paddle can move at least `WIDTH - 1` times, which is enough to go from any
    position to any other one. So the optimal policy (move towards the target until it is right below it, then stay still)
    always catches the ball: the maximal mean reward is 1.0.
    """)
    return


@app.cell
def _(F, HEIGHT, WIDTH, torch):
    def max_mean_reward():
        "Mean reward of the optimal strategy"
        return 1.0

    def sample_position(shape):
        "Random position values in {0, ..., WIDTH-1}, mapped into [-1.0, 1.0]"
        i = torch.randint(0, WIDTH, shape).float()
        return 2.0 * i / (WIDTH - 1) - 1.0

    def sample_velocity(shape):
        "Random velocity values in {-1.0, 0.0, 1.0}"
        return torch.randint(-1, 2, shape).float()

    def model_input(x, target_x, target_dx):
        "Paddle positions, target positions and target velocity, stacked as the 3 model inputs"
        return torch.cat((x, target_x, target_dx), dim=-1)

    def rebound_in_place(target_x, target_dx):
        right = target_x > 1.0
        target_x[right] = 2.0 - target_x[right]
        target_dx[right] = - target_dx[right]
        left = target_x < -1.0
        target_x[left] = -2.0 - target_x[left]
        target_dx[left] = - target_dx[left]

    def step(x, target_x, target_dx, u):
        "Paddle and ball position after the move u in {-1, 0, 1}"
        x_next = x + 2.0 * u / (WIDTH - 1.0)
        x_next = x_next.clamp(-1.0, 1.0)
        target_dx_next = target_dx.clone()
        target_x_next = target_x + target_dx_next * 2.0 / (WIDTH - 1.0)
        rebound_in_place(target_x_next, target_dx_next)
        return x_next, target_x_next, target_dx_next

    def reward(x, target_x):
        return torch.isclose(x, target_x).float()

    def mean_reward(model, num_samples=1_000):
        with torch.no_grad():
            x = sample_position((num_samples, 1))
            target_x = sample_position((num_samples, 1))
            target_dx = sample_velocity((num_samples, 1))
            for _t in range(HEIGHT - 1):
                input = model_input(x, target_x, target_dx)
                probas = F.softmax(model(input), dim=-1)
                u = torch.multinomial(probas, num_samples=1) - 1.0
                x, target_x, target_dx = step(x, target_x, target_dx, u)
            return reward(x, target_x).mean().item()

    return (
        max_mean_reward,
        mean_reward,
        model_input,
        reward,
        sample_position,
        sample_velocity,
        step,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Same method as in stage 2, gradient ascent of the mean reward, but the reward now depends on the whole
    sequence of actions $u_0, \dots, u_{H-2}$ (with $H = {\rm HEIGHT}$). The REINFORCE trick still applies;
    the log-probability of a trajectory is the sum of the log-probabilities of its actions:

    $$
    \nabla_{\theta} \mathbb{E}[r] = \mathbb{E}\left[r \, \sum_{t=0}^{H-2} \nabla_{\theta} \log \pi_{\theta}(u_t \mid x_t, \texttt{target\_x}, \texttt{target\_dx})\right]
    $$

    So we first simulate the trajectories (without gradient tracking), then sum the log-probabilities of the actions
    that were taken.
    """)
    return


@app.cell
def _(
    F,
    HEIGHT,
    model_input,
    reward,
    sample_position,
    sample_velocity,
    step,
    torch,
):
    def mean_reward_grad(model, num_samples=1_000):
        """
        Estimate the gradient of the mean reward wrt model weights
        using the REINFORCE log-derivative trick and sampling.
        The result is stored in the `grad` attribute of the model parameters.
        """
        with torch.no_grad():
            x = sample_position((num_samples, 1))
            target_x = sample_position((num_samples, 1))
            target_dx = sample_velocity((num_samples, 1))
            history = [(x, target_x, target_dx)]
            indices = []
            for _t in range(HEIGHT - 1):
                input = model_input(x, target_x, target_dx)
                probas = F.softmax(model(input), dim=-1)
                index = torch.multinomial(probas, num_samples=1)
                indices.append(index)
                u = index - 1
                x, target_x, target_dx = step(x, target_x, target_dx, u)
                history.append((x, target_x, target_dx))
            r = reward(x, target_x)

        log_probs_sum = 0.0
        for t in range(HEIGHT - 1):
            input = model_input(*history[t])
            log_probs = F.log_softmax(model(input), dim=-1)
            log_probs_sum = log_probs_sum + log_probs.gather(dim=1, index=indices[t])

        value = (r * log_probs_sum).mean()
        model.zero_grad()
        value.backward()

    return (mean_reward_grad,)


@app.cell
def _(
    DecisionModel,
    max_mean_reward,
    mean_reward,
    mean_reward_grad,
    mo,
    plt,
    torch,
):
    def train_model(n=10_000):
        policy = DecisionModel()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2, maximize=True)
        rewards = [mean_reward(policy)]
        for _ in range(n):
            mean_reward_grad(policy)
            optimizer.step()
            rewards.append(mean_reward(policy))
        return policy, rewards

    policy, rewards = train_model()
    torch.save(policy.state_dict(), "models/model07.pt")

    plt.title("Mean reward during training")
    plt.xlabel("step")
    plt.plot(rewards)
    plt.axhline(max_mean_reward(), color="black", linestyle="--", label=f"max mean reward ({max_mean_reward():.2f})")
    plt.legend()
    plt.grid(True)
    mo.center(plt.gcf())
    return (policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The policy depends on the paddle and target positions $(x_t, \texttt{target\_x})$ and on the target
    velocity $\texttt{target\_dx}$, so we show heatmaps over the $(\texttt{target\_x}, x_t)$ plane, one per action
    (columns) and per target velocity (rows), for the action probabilities.
    """)
    return


@app.cell
def _(max_mean_reward, mean_reward, mo, model_input, plt, policy, torch):
    def _(num_points=100):
        with torch.inference_mode():
            x_t = torch.linspace(-1.0, 1.0, num_points)
            target_x = torch.linspace(-1.0, 1.0, num_points)
            Xt, TargetX = torch.meshgrid(x_t, target_x, indexing="ij")
            Xt, TargetX = Xt.reshape(-1, 1), TargetX.reshape(-1, 1)
            p = [
                policy(model_input(Xt, TargetX, torch.full_like(Xt, dx)))
                .softmax(dim=-1)
                .reshape(num_points, num_points, 3)
                for dx in (-1.0, 0.0, 1.0)
            ]

        labels = ["move left", "stay still", "move right"]
        fig, axes = plt.subplots(3, 3, figsize=(12, 11), sharex=True, sharey=True)
        for i, dx in enumerate((-1.0, 0.0, 1.0)):
            for j, ax in enumerate(axes[i]):
                im = ax.imshow(p[i][:, :, j], origin="lower", extent=[-1, 1, -1, 1],
                               vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
                ax.set_title(f"{labels[j]} (target_dx = {dx:+.0f})")
                if i == 2:
                    ax.set_xlabel("target_x")
                if j == 0:
                    ax.set_ylabel("x_t")
                fig.colorbar(im, ax=ax)
        fig.suptitle(f"Mean reward: {mean_reward(policy):.3f} (max: {max_mean_reward():.3f})")
        fig.tight_layout()
        return mo.center(fig)

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulation
    """)
    return


@app.cell
def _(
    HEIGHT,
    model_input,
    pd,
    policy,
    reward,
    sample_position,
    sample_velocity,
    step,
    torch,
):
    def _():
        num_samples = 100  # batch size
        x_0 = sample_position((num_samples, 1))
        target_x_0 = sample_position((num_samples, 1))
        target_dx_0 = sample_velocity((num_samples, 1))
        x_t, target_x, target_dx = x_0, target_x_0, target_dx_0
        with torch.inference_mode():
            for _t in range(HEIGHT - 1):
                input = model_input(x_t, target_x, target_dx)
                u_t = policy(input).argmax(dim=1, keepdim=True) - 1.0  # deterministic choice
                x_t, target_x, target_dx = step(x_t, target_x, target_dx, u_t)
        df = pd.DataFrame(
            {
                "target_x (initial)": target_x_0.squeeze(1),
                "target_dx (initial)": target_dx_0.squeeze(1),
                "target_x (final)": target_x.squeeze(1),
                "paddle_x (initial)": x_0.squeeze(1),
                "paddle_x (final)": x_t.squeeze(1),
                "success": reward(x_t, target_x).squeeze(1).bool(),
            }
        )
        return df

    _()
    return


if __name__ == "__main__":
    app.run()
