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
    # Breakout game, stage 1

    Same game as in stage 0: a grid of width 3 and height 2, a ball dropped at a random top cell (left, center or right)
    and a paddle that starts at the center of the bottom row and may move left, stay still or move right to catch it.

    We use the same decision model as in stage 0, a linear layer with:

    - 1 input : the location of the ball, between -1.0 (left) and 1.0 (right)
    - 3 outputs : the logit of every possible action (move left, stay still, move right)

    But this time, instead of setting its weights by hand, we **train** it.
    """)
    return


@app.cell
def _(torch):
    def DecisionModel():
        return torch.nn.Linear(in_features=1, out_features=3)

    return (DecisionModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reward

    The paddle starts at the center ($x=0$). The reward is 1 if the paddle catches the ball ($u = x_{\rm ref}$) and 0 otherwise.
    The actions are sampled from the policy (the softmax of the logits), so the mean reward is estimated by sampling.
    """)
    return


@app.cell
def _(F, torch):
    def sample_target_x(shape):
        "Random values in {-1, 0, 1} (as floats)"
        return torch.randint(-1, 2, shape).float()

    def reward(target_x, u):
        paddle_x = u  # since x = x0 + u and x0 = 0
        return (paddle_x == target_x).float()

    def mean_reward(model, num_samples=1_000):
        with torch.no_grad():
            target_x = sample_target_x((num_samples, 1))
            probas = F.softmax(model(target_x), dim=-1)
            u = torch.multinomial(probas, num_samples=1) - 1.0
            return reward(target_x, u).mean().item()

    return mean_reward, reward, sample_target_x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    We maximize the mean reward with gradient ascent. Since the actions are sampled, the reward is not differentiable
    with respect to the model weights; we estimate its gradient with the REINFORCE (log-derivative) trick:

    $$
    \nabla_{\theta} \mathbb{E}[r] = \mathbb{E}[r \, \nabla_{\theta} \log \pi_{\theta}({\rm action} \mid {\rm target})]
    $$
    """)
    return


@app.cell
def _(F, reward, sample_target_x, torch):
    def mean_reward_grad(model, num_samples=1_000):
        """
        Estimate the gradient of the mean reward wrt model weights
        using the REINFORCE log-derivative trick and sampling.
        The result is stored in the `grad` attribute of the model parameters.
        """
        target_x = sample_target_x((num_samples, 1))  # batch of target positions
        log_probs = F.log_softmax(model(target_x), dim=-1)
        with torch.no_grad():
            index = torch.multinomial(log_probs.exp(), num_samples=1)
            u = index - 1.0
            r = reward(target_x, u)
        selected_log_probs = log_probs.gather(dim=1, index=index)
        value = (r * selected_log_probs).mean()
        model.zero_grad()
        value.backward()

    return (mean_reward_grad,)


@app.cell
def _(DecisionModel, mean_reward, mean_reward_grad, mo, plt, torch):
    def train_model(n=50_000):
        policy = DecisionModel()
        optimizer = torch.optim.Adam(policy.parameters(), maximize=True)
        rewards = [mean_reward(policy)]
        for _ in range(n):
            mean_reward_grad(policy)
            optimizer.step()
            rewards.append(mean_reward(policy))
        return policy, rewards

    policy, rewards = train_model()
    torch.save(policy.state_dict(), "models/model01.pt")

    plt.title("Mean reward during training")
    plt.xlabel("step")
    plt.plot(rewards)
    plt.grid(True)
    mo.center(plt.gcf())


    return (policy,)


@app.cell
def _(DecisionModel, mo, plt, policy, torch):
    policy

    def _():
        policy = DecisionModel()
        policy.load_state_dict(torch.load("models/model01.pt"))

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
        state_dict = torch.load("models/model01.pt")
        policy.load_state_dict(state_dict)

        num_samples = 100 # batch size
        target_x = torch.randint(-1, 2, (num_samples, 1)).float()
        with torch.inference_mode():
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
