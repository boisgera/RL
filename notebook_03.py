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
    # Breakout game, stage 3

    The game now takes place on a grid of width `WIDTH` and height `HEIGHT` (larger than 2).
    The paddle starts at a random (uniform) position on the lower level and the ball pops at a random (uniform)
    position on the upper level, independently. The ball drops one level at every tick, so it reaches the lower level
    after `HEIGHT - 1` ticks; at every tick, the paddle can decide to stay at its position, move left or move right
    (within the grid). Therefore the paddle may move `HEIGHT - 1` times to try and catch the ball.

    The paddle and target positions are integers in $\{0, \dots, {\rm WIDTH}-1\}$, always mapped linearly to $[-1.0, 1.0]$.
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
    The decision model is the same as in stage 2, a linear layer with no hidden layer, used at every tick:

    - 2 inputs : the current paddle position $x_t$ and the target position $\texttt{target\_x}$, between -1.0 (left) and 1.0 (right)
    - 3 outputs : the logit of every possible action (move left, stay still, move right)
    """)
    return


@app.cell
def _(torch):
    def DecisionModel():
        return torch.nn.Linear(in_features=2, out_features=3)

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

    def model_input(x_t, target_x):
        "Paddle and target positions, stacked as the 2 model inputs"
        return torch.cat((x_t, target_x), dim=-1)

    def step(x_t, u):
        "Paddle position after the move u in {-1, 0, 1}"
        x_next = x_t + 2.0 * u / (WIDTH - 1.0)
        return x_next.clamp(-1.0, 1.0)

    def reward(x, target_x):
        return torch.isclose(x, target_x).float()

    def mean_reward(model, num_samples=1_000):
        with torch.no_grad():
            x_t = sample_position((num_samples, 1))
            target_x = sample_position((num_samples, 1))
            for _t in range(HEIGHT - 1):
                probas = F.softmax(model(model_input(x_t, target_x)), dim=-1)
                u_t = torch.multinomial(probas, num_samples=1) - 1.0
                x_t = step(x_t, u_t)
            return reward(x_t, target_x).mean().item()

    return max_mean_reward, mean_reward, model_input, reward, sample_position, step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Same method as in stage 2, gradient ascent of the mean reward, but the reward now depends on the whole
    sequence of actions $u_0, \dots, u_{H-2}$ (with $H = {\rm HEIGHT}$). The REINFORCE trick still applies;
    the log-probability of a trajectory is the sum of the log-probabilities of its actions:

    $$
    \nabla_{\theta} \mathbb{E}[r] = \mathbb{E}\left[r \, \sum_{t=0}^{H-2} \nabla_{\theta} \log \pi_{\theta}(u_t \mid x_t, \texttt{target\_x})\right]
    $$

    So we first simulate the trajectories (without gradient tracking), then sum the log-probabilities of the actions
    that were taken.
    """)
    return


@app.cell
def _(F, HEIGHT, model_input, reward, sample_position, step, torch):
    def mean_reward_grad(model, num_samples=1_000):
        """
        Estimate the gradient of the mean reward wrt model weights
        using the REINFORCE log-derivative trick and sampling.
        The result is stored in the `grad` attribute of the model parameters.
        """
        with torch.no_grad():
            x_0 = sample_position((num_samples, 1))
            target_x = sample_position((num_samples, 1))
            x = [x_0]
            indices = []
            for _t in range(HEIGHT - 1):
                probas = F.softmax(model(model_input(x[-1], target_x)), dim=-1)
                index = torch.multinomial(probas, num_samples=1)
                indices.append(index)
                x.append(step(x[-1], index - 1.0))
            r = reward(x[-1], target_x)

        log_probs_sum = 0.0
        for t in range(HEIGHT - 1):
            log_probs = F.log_softmax(model(model_input(x[t], target_x)), dim=-1)
            log_probs_sum = log_probs_sum + log_probs.gather(dim=1, index=indices[t])

        value = (r * log_probs_sum).mean()
        model.zero_grad()
        value.backward()

    return (mean_reward_grad,)


@app.cell
def _(DecisionModel, max_mean_reward, mean_reward, mean_reward_grad, mo, plt, torch):
    def train_model(n=5_000):
        policy = DecisionModel()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2, maximize=True)
        rewards = [mean_reward(policy)]
        for _ in range(n):
            mean_reward_grad(policy)
            optimizer.step()
            rewards.append(mean_reward(policy))
        return policy, rewards

    policy, rewards = train_model()
    torch.save(policy.state_dict(), "models/model03.pt")

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
    The policy depends on the paddle and target positions $(x_t, \texttt{target\_x})$, so we show heatmaps
    over the $(\texttt{target\_x}, x_t)$ plane, one per action, for the action probabilities.
    """)
    return


@app.cell
def _(max_mean_reward, mean_reward, mo, model_input, plt, policy, torch):
    def _(num_points=100):
        with torch.inference_mode():
            x_t = torch.linspace(-1.0, 1.0, num_points)
            target_x = torch.linspace(-1.0, 1.0, num_points)
            Xt, TargetX = torch.meshgrid(x_t, target_x, indexing="ij")
            logits = policy(model_input(Xt.reshape(-1, 1), TargetX.reshape(-1, 1)))
        p = logits.softmax(dim=-1).reshape(num_points, num_points, 3)

        labels = ["move left", "stay still", "move right"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for j, ax in enumerate(axes):
            im = ax.imshow(p[:, :, j], origin="lower", extent=[-1, 1, -1, 1],
                           vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
            ax.set_title(labels[j])
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
def _(HEIGHT, model_input, pd, policy, reward, sample_position, step, torch):
    def _():
        num_samples = 100  # batch size
        x_0 = sample_position((num_samples, 1))
        target_x = sample_position((num_samples, 1))
        x_t = x_0
        with torch.inference_mode():
            for _t in range(HEIGHT - 1):
                u_t = policy(model_input(x_t, target_x)).argmax(dim=1, keepdim=True) - 1.0  # deterministic choice
                x_t = step(x_t, u_t)
        df = pd.DataFrame(
            {
                "target_x": target_x.squeeze(1),
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
