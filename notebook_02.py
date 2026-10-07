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
    # Breakout game, stage 2

    The game takes place on a grid of width `WIDTH` (larger than 3) and height 2. The paddle starts at a random (uniform)
    position on the lower level and the ball pops at a random (uniform) position on the upper level, independently;
    it will drop vertically at the next tick. The paddle can decide to stay at its position, move left or move right
    (within the grid) for the next tick to try and catch the ball.

    The dynamics still resolves in one step, which means that in many cases, no policy can catch the ball,
    especially when the width is large.

    The paddle and target positions are integers in $\{0, \dots, {\rm WIDTH}-1\}$, always mapped linearly to $[-1.0, 1.0]$.
    """)
    return


@app.cell
def _():
    WIDTH = 5
    return (WIDTH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The decision model is still a linear layer with no hidden layer, but it now has 2 inputs,
    since the paddle position is random too:

    - 2 inputs : the paddle position $x_0$ and the target position $x_{\rm ref}$, between -1.0 (left) and 1.0 (right)
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

    The reward is 1 if the paddle catches the ball and 0 otherwise.

    For every target position except on the boundary, there are only three initial positions of the paddle,
    in the neighbourhood of the target, where a movement allows to catch the ball. On the boundary, there are two.
    So there are `(WIDTH - 2) * 3 + 2 * 2` cases where the reward can be 1, otherwise that's 0.
    Since there are `WIDTH * WIDTH` possible combinations of the target position and paddle position,
    the maximal mean reward is:

        ((WIDTH - 2) * 3 + 2 * 2) / (WIDTH * WIDTH)
    """)
    return


@app.cell
def _(F, WIDTH, torch):
    def max_mean_reward():
        "Mean reward of the optimal strategy"
        return ((WIDTH - 2) * 3 + 2 * 2) / (WIDTH * WIDTH)

    def sample_position(shape):
        "Random position values in {0, ..., WIDTH-1}, mapped into [-1.0, 1.0]"
        i = torch.randint(0, WIDTH, shape).float()
        return 2.0 * i / (WIDTH - 1) - 1.0

    def model_input(x_0, x_ref):
        "Paddle and target positions, stacked as the 2 model inputs"
        return torch.cat((x_0, x_ref), dim=-1)

    def step(x_0, u):
        "Paddle position after the move u in {-1, 0, 1}"
        x = x_0 + 2.0 * u / (WIDTH - 1.0)
        return x.clamp(-1.0, 1.0)

    def reward(x_0, x_ref, u):
        return torch.isclose(step(x_0, u), x_ref).float()

    def mean_reward(model, num_samples=1_000):
        with torch.no_grad():
            x_0 = sample_position((num_samples, 1))
            x_ref = sample_position((num_samples, 1))
            probas = F.softmax(model(model_input(x_0, x_ref)), dim=-1)
            u = torch.multinomial(probas, num_samples=1) - 1.0
            return reward(x_0, x_ref, u).mean().item()

    return max_mean_reward, mean_reward, model_input, reward, sample_position, step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Same method as in stage 1: gradient ascent of the mean reward, whose gradient is estimated with the REINFORCE trick:

    $$
    \nabla_{\theta} \mathbb{E}[r] = \mathbb{E}[r \, \nabla_{\theta} \log \pi_{\theta}({\rm action} \mid {\rm target}, {\rm paddle})]
    $$
    """)
    return


@app.cell
def _(F, model_input, reward, sample_position, torch):
    def mean_reward_grad(model, num_samples=1_000):
        """
        Estimate the gradient of the mean reward wrt model weights
        using the REINFORCE log-derivative trick and sampling.
        The result is stored in the `grad` attribute of the model parameters.
        """
        x_0 = sample_position((num_samples, 1))
        x_ref = sample_position((num_samples, 1))
        log_probs = F.log_softmax(model(model_input(x_0, x_ref)), dim=-1)
        with torch.no_grad():
            index = torch.multinomial(log_probs.exp(), num_samples=1)
            u = index - 1.0
            r = reward(x_0, x_ref, u)
        selected_log_probs = log_probs.gather(dim=1, index=index)
        value = (r * selected_log_probs).mean()
        model.zero_grad()
        value.backward()

    return (mean_reward_grad,)


@app.cell
def _(DecisionModel, max_mean_reward, mean_reward, mean_reward_grad, mo, plt, torch):
    def train_model(n=10_000):
        policy = DecisionModel()
        optimizer = torch.optim.Adam(policy.parameters(), maximize=True)
        rewards = [mean_reward(policy)]
        for _ in range(n):
            mean_reward_grad(policy)
            optimizer.step()
            rewards.append(mean_reward(policy))
        return policy, rewards

    policy, rewards = train_model()
    torch.save(policy.state_dict(), "models/model02.pt")

    plt.title("Mean reward during training")
    plt.xlabel("step")
    plt.plot(rewards)
    plt.axhline(max_mean_reward(), color="black", linestyle="--", label="max mean reward")
    plt.legend()
    plt.grid(True)
    mo.center(plt.gcf())
    return (policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The policy depends on the paddle and target positions $(x_0, x_{\rm ref})$, so we show heatmaps
    over the $(x_{\rm ref}, x_0)$ plane, one per action, for the action probabilities.
    """)
    return


@app.cell
def _(max_mean_reward, mean_reward, mo, model_input, plt, policy, torch):
    def _(num_points=100):
        with torch.inference_mode():
            x_0 = torch.linspace(-1.0, 1.0, num_points)
            x_ref = torch.linspace(-1.0, 1.0, num_points)
            X0, Xref = torch.meshgrid(x_0, x_ref, indexing="ij")
            logits = policy(model_input(X0.reshape(-1, 1), Xref.reshape(-1, 1)))
        p = logits.softmax(dim=-1).reshape(num_points, num_points, 3)

        labels = ["move left", "stay still", "move right"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        for j, ax in enumerate(axes):
            im = ax.imshow(p[:, :, j], origin="lower", extent=[-1, 1, -1, 1],
                           vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
            ax.set_title(labels[j])
            ax.set_xlabel("x_ref")
            if j == 0:
                ax.set_ylabel("x_0")
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
def _(model_input, pd, policy, reward, sample_position, step, torch):
    def _():
        num_samples = 100  # batch size
        x_0 = sample_position((num_samples, 1))
        x_ref = sample_position((num_samples, 1))
        with torch.inference_mode():
            action_logits = policy(model_input(x_0, x_ref))
        u = action_logits.argmax(dim=1, keepdim=True) - 1.0  # deterministic choice
        df = pd.DataFrame(
            {
                "target_x": x_ref.squeeze(1),
                "paddle_x (initial)": x_0.squeeze(1),
                "move": u.squeeze(1),
                "paddle_x (final)": step(x_0, u).squeeze(1),
                "success": reward(x_0, x_ref, u).squeeze(1).bool(),
            }
        )
        return df

    _()
    return


if __name__ == "__main__":
    app.run()
