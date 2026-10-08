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
    # Breakout game, stage 8

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
    The decision model is the same as in stage 7: two hidden layers of 32 units each (with ReLU activation),
    used at every tick:

    - 3 inputs : the current paddle position $x_t$ and target position $\texttt{target\_x}$, between -1.0 (left) and 1.0 (right), and the target velocity $\texttt{target\_dx}$ in $\{-1.0, 0.0, 1.0\}$
    - 3 outputs : the logit of every possible action (move left, stay still, move right)

    In stage 7, the training of this network got stuck on a plateau (mean reward around 0.60 to 0.70), and the
    entropy of the policy showed a premature collapse: the policy became almost deterministic long before it
    reached the maximal mean reward, and then stopped exploring. In this stage, we add an entropy bonus to the
    objective to prevent this collapse.
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

    **Entropy bonus.** To keep the policy from collapsing prematurely into a deterministic one, we now maximize
    the mean reward plus a bonus proportional to the mean entropy of the policy over the visited states:

    $$
    J(\theta) = \mathbb{E}[r] + \beta \, \mathbb{E}\left[\frac{1}{H-1} \sum_{t=0}^{H-2} \mathcal{H}(\pi_{\theta}(\cdot \mid x_t, \texttt{target\_x}, \texttt{target\_dx}))\right]
    $$

    where $\mathcal{H}(p) = - \sum_u p(u) \log p(u)$ and $\beta = \texttt{ENTROPY\_BONUS}$. The entropy of the
    action distribution in a given state is an explicit function of the model output, so we differentiate it directly
    in the visited states (we neglect the influence of $\theta$ on the distribution of these states).
    """)
    return


@app.cell
def _():
    ENTROPY_BONUS = 0.3
    return (ENTROPY_BONUS,)


@app.cell
def _(
    ENTROPY_BONUS,
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
        Estimate the gradient of the mean reward (plus entropy bonus) wrt model weights
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
        entropy_sum = 0.0
        for t in range(HEIGHT - 1):
            input = model_input(*history[t])
            log_probs = F.log_softmax(model(input), dim=-1)
            log_probs_sum = log_probs_sum + log_probs.gather(dim=1, index=indices[t])
            entropy_sum = entropy_sum - (log_probs.exp() * log_probs).sum(dim=-1, keepdim=True)

        value = (r * log_probs_sum + ENTROPY_BONUS * entropy_sum / (HEIGHT - 1)).mean()
        model.zero_grad()
        value.backward()

    return (mean_reward_grad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To detect a premature collapse of the policy (which would stop the exploration), we also monitor
    during training the mean entropy of the action distribution over the states visited by the policy:

    $$
    H(\pi_{\theta}(\cdot \mid s)) = - \sum_{u \in \{-1, 0, 1\}} \pi_{\theta}(u \mid s) \log \pi_{\theta}(u \mid s)
    $$

    With 3 possible actions, the entropy is between 0 (deterministic policy) and $\log 3 \approx 1.099$
    (uniform policy).
    """)
    return


@app.cell
def _(F, HEIGHT, model_input, sample_position, sample_velocity, step, torch):
    def max_entropy():
        "Entropy of the uniform distribution over the 3 actions"
        return torch.log(torch.tensor(3.0)).item()

    def mean_entropy(model, num_samples=1_000):
        "Mean entropy of the action distribution over the states visited by the policy"
        with torch.no_grad():
            x = sample_position((num_samples, 1))
            target_x = sample_position((num_samples, 1))
            target_dx = sample_velocity((num_samples, 1))
            entropies = []
            for _t in range(HEIGHT - 1):
                input = model_input(x, target_x, target_dx)
                log_probas = F.log_softmax(model(input), dim=-1)
                probas = log_probas.exp()
                entropies.append(-(probas * log_probas).sum(dim=-1))
                u = torch.multinomial(probas, num_samples=1) - 1.0
                x, target_x, target_dx = step(x, target_x, target_dx, u)
            return torch.cat(entropies).mean().item()

    return max_entropy, mean_entropy


@app.cell
def _(
    DecisionModel,
    max_entropy,
    max_mean_reward,
    mean_entropy,
    mean_reward,
    mean_reward_grad,
    mo,
    plt,
    torch,
):
    def train_model(n=1_000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        policy = DecisionModel()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3, maximize=True)
        rewards = [mean_reward(policy)]
        entropies = [mean_entropy(policy)]
        for _ in range(n):
            mean_reward_grad(policy, num_samples=1_000)
            optimizer.step()
            rewards.append(mean_reward(policy, num_samples=1_000))
            entropies.append(mean_entropy(policy, num_samples=1_000))
        return policy, rewards, entropies

    policy, rewards, entropies = train_model(n=2_000, seed=0)
    torch.save(policy.state_dict(), "models/model08.pt")
    fig, (ax_reward, ax_entropy) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_reward.set_title("Mean reward during training")
    ax_reward.plot(rewards, label="mean reward", color='C0', alpha=0.8)
    final_reward = rewards[-1]
    ax_reward.axhline(final_reward, color="C0", linestyle="--", label=f"final mean reward ({final_reward:.2f})")
    ax_reward.axhline(max_mean_reward(), color="black", linestyle="--", label=f"max mean reward ({max_mean_reward():.2f})")
    ax_reward.legend()
    ax_reward.grid(True)
    ax_entropy.set_title("Mean policy entropy during training")
    ax_entropy.set_xlabel("step")
    ax_entropy.plot(entropies, label="mean entropy", color='C1', alpha=0.8)
    ax_entropy.axhline(max_entropy(), color="black", linestyle="--", label=f"max entropy (log 3 = {max_entropy():.3f})")
    ax_entropy.set_ylim(0.0, 1.1 * max_entropy())
    ax_entropy.legend()
    ax_entropy.grid(True)
    fig.tight_layout()
    mo.center(fig)
    return (policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The entropy bonus fixes the premature collapse. With $\beta = 0.3$ and 10 different random seeds, the
    deterministic (argmax) policy catches the ball in all 75 initial states after 1000 steps, and the mean
    reward of the sampled policy is between 0.97 and 0.99 after 2000 steps. The entropy decreases more slowly
    and then stays around 0.17 to 0.21 instead of going to 0: the policy keeps a little randomness, which is why
    the sampled mean reward stays slightly below 1.0.

    The value of $\beta$ matters (3 seeds each, 5000 steps, unless stated otherwise):

    | $\beta$ | mean reward (sampled) | initial states caught (argmax) | final entropy |
    |---|---|---|---|
    | 0 (stage 7) | 0.62 to 0.65 | 47 / 75 | ~0.01 |
    | 0.01, 0.03 | 0.62 to 0.65 | 47 / 75 | 0.01 to 0.04 |
    | 0.1 | 0.70 to 0.86 | 52 to 65 / 75 | 0.10 to 0.15 |
    | 0.2 (10 seeds, 2000 steps) | 0.82 to 0.995 | 61 to 75 / 75 (5 seeds out of 10 reach 75) | 0.14 to 0.21 |
    | 0.3 (10 seeds, 2000 steps) | 0.97 to 0.99 | 75 / 75 | 0.17 to 0.21 |
    | 0.5 (10 seeds, 2000 steps) | 0.94 to 0.95 | 75 / 75 | 0.31 to 0.37 |
    | 1.0 | 0.71 to 0.73 | 75 / 75 (one seed: 74) | ~0.75 |

    A small bonus does not prevent the collapse. A large bonus finds the right decisions (the argmax policy is
    optimal) but keeps the policy too random, so the sampled mean reward is lower.
    """)
    return


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
