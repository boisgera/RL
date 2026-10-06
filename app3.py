# Python Standard Library
import itertools

# Third-Party Libraries
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


# TODO: adjustment: now the game has a HEIGHT is larger than 1,
# therefore the paddle may move HEIGHT - 1 times to try and catch the ball.


WIDTH = 5
HEIGHT = 5
assert HEIGHT >= WIDTH  # Simpler: we know that we can always catch the ball.

"""
We consider a "mini-breakout" game where the paddle should catch the ball.
Everything takes place on a WIDTH-cell grid, at a lower level (the paddle)
and an upper level (the ball). The x-coordinate is measured on this grid as
an integer position in {0, ..., WIDTH-1}, linearly mapped to [-1.0, 1.0];
the paddle starts at a random (uniform) position and the ball pops at the
upper level at a random (uniform) position, and will drop vertically at
next tick. The paddle can decide to stay at its position, move left or
move right for the next tick to try and catch the ball.
"""

def max_mean_reward():
    """
    Compute the reward for the optimal strategy.
    """
    return 1.0

def sample_position(num_samples=1):
    "Random position values in {0, ..., WIDTH-1}, mapped into [-1.0, 1.0]"
    i = torch.randint(0, WIDTH, (num_samples,)).float()
    return 2.0 * i / (WIDTH - 1) - 1.0

def Model(hidden_sizes=()):
    """
    - input: paddle position and target position in [-1, 1], 
    - output: action policy (3 logits). 
    
    hidden_sizes: size of each hidden layer, e.g. [] or [16] or [16, 16].
    """
    sizes = [2, *hidden_sizes, 3]
    layers = []
    for i, (in_size, out_size) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(in_size, out_size))
        if i < len(hidden_sizes):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def sample_u(policy, num_samples=1):
    """
    Random values of the control in {-1, 0, 1}

    The policy is the law of the control, as a tensor of 3 unnormalized log probas
    """
    u_values = torch.tensor([-1, 0, 1])
    probas = F.softmax(policy, dim=-1)
    i = torch.multinomial(probas, num_samples=1)    
    u = u_values[i]
    return u

def step(x, u):
    x_next = x + 2.0 * u / (WIDTH - 1.0)
    x_next = x_next.clamp(-1.0, 1.0)    
    return x_next

def reward(x, x_ref, u):
    return (x == x_ref).float()

def mean_reward(model, num_samples=1_000):
    n = num_samples
    with torch.no_grad():
        x_t = sample_position(n)
        x_ref = sample_position(n)
        for _t in range(HEIGHT - 1):
            state = torch.stack((x_t, x_ref), dim=1)
            policy = model(state)
            u_t = sample_u(policy, n).squeeze(1)
            x_t = step(x_t, u_t)
        r = reward(x_t, x_ref, u_t)
    return r.mean().item()

def mean_reward_grad(model, num_samples=1_000):
    """
    Estimate the gradient of the mean reward wrt model weights 
    using the REINFORCE log-derivative trick and sampling.
    The result is stored in `model.grad`.
    """
    n = num_samples
    with torch.no_grad():
        x_0 = sample_position(n)
        x_ref = sample_position(n)
        u_choices = torch.tensor([-1, 0, 1])
        x = [x_0]
        indices = []
        for _t in range(HEIGHT - 1):
            x_t = x[-1]
            state = torch.stack((x_t, x_ref), dim=1)    
            logits = model(state)
            probs = logits.softmax(dim=-1)
            index = torch.multinomial(probs, num_samples=1)
            indices.append(index)
            u_t = u_choices[index].squeeze(1)
            x_t = step(x[-1], u_t) 
            x.append(x_t)
    r = reward(x[-1], x_ref, u_t)

    log_probs_sum = 0.0
    for t in range(HEIGHT - 1):
        state = torch.stack((x[t], x_ref), dim=1)
        logits = model(state)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_sum = log_probs_sum + log_probs.gather(dim=1, index=indices[t]).squeeze(1)

    value = (r * log_probs_sum).mean()
    model.zero_grad()
    value.backward()

def plot(model, num_points=100):
    """
    The policy now depends on two inputs (x_0, x_ref), so instead of curves
    against x_ref alone, we show heatmaps over the (x_ref, x_0) plane, one
    per action, for the action probabilities.
    """
    with torch.inference_mode():
        x_0 = torch.linspace(-1.0, 1.0, num_points)
        x_ref = torch.linspace(-1.0, 1.0, num_points)
        X0, Xref = torch.meshgrid(x_0, x_ref, indexing="ij")
        inputs = torch.stack((X0.reshape(-1), Xref.reshape(-1)), dim=1)
        logits = model(inputs)
    logits = logits.detach()
    p = logits.softmax(dim=-1)
    p = p.reshape(num_points, num_points, 3)

    labels = ["-1", "0", "+1"]
    extent = [-1, 1, -1, 1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for j in range(3):
        ax = axes[j]
        im = ax.imshow(p[:, :, j], origin="lower", extent=extent,
                        vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
        ax.set_title(rf"$\mathbb{{P}}(u={labels[j]})$")
        ax.set_xlabel("x_ref")
        if j == 0:
            ax.set_ylabel("x_0")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Mean reward: {mean_reward(model):.3f} "
                 f"(max: {max_mean_reward():.3f})")
    fig.tight_layout()

def train(model, steps=5_000):
    optimizer = optim.Adam(model.parameters(), maximize=True)
    print(f"Step 00 -- reward: {mean_reward(model)}")
    for i in range(steps):
        mean_reward_grad(model)
        optimizer.step()
        print(f"Step {i+1:2} -- reward: {mean_reward(model)}")


if __name__ == "__main__":
    model = Model([16])
    train(model, steps=10_000)
    plot(model)
    plt.show()
    
