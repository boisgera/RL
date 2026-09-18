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


# TODO: adjustment: the width of the game is larger and the paddle initial
# location is also random. But the dynamics still resolves in one step
# (which means that for many cases, there is no policy that can achieve
# the goal, especially when the width is large).

# Note: the paddle location and target location are always mapped linearly
# from [0, ... WIDTH-1] to [-1.0, 1.0] before being entered into the policy
# model.

WIDTH = 5

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

    In the general case (WIDTH >= 2), for every target position 
    except on the boundary, there are only three initial positions of the pad,
    in the neighbourhood of the target, where a movement allows to catch the ball.
    On the boundary, there are two. So `(WIDTH - 2) * 3 + 2 * 2` cases
    where the reward can be 1, otherwise that's 0.
    Since there are WIDTH^2 possible combinations of the target position and 
    paddle position, the maximal mean reward is:

        ((WIDTH - 2) * 3 + 2 * 2) / (WIDTH * WIDTH)
    
    Now, "by chance", the formula also work when `WIDTH = 1`; in this
    case the maximal mean reward is `1`.
    """
    return ((WIDTH - 2) * 3 + 2 * 2) / (WIDTH * WIDTH)

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

def reward(x_0, x_ref, u):
    x = x_0 + 2.0 * u / (WIDTH - 1.0)
    x = x.clamp(-1.0, 1.0)
    return (x == x_ref).float()

def mean_reward(model, num_samples=1_000):
    x_0 = sample_position(num_samples)
    x_ref = sample_position(num_samples)
    input = torch.stack((x_0, x_ref), dim=1)
    policy = model(input)
    u = sample_u(policy, num_samples).squeeze(1)
    r = reward(x_0, x_ref, u)
    return r.mean().item()

def mean_reward_grad(model, num_samples=1_000):
    """
    Estimate the gradient of the mean reward wrt model weights 
    using the REINFORCE log-derivative trick and sampling.
    The result is stored in `model.grad`.
    """
    n = num_samples
    x_0 = sample_position(n)
    x_ref = sample_position(n)
    input = torch.stack((x_0, x_ref), dim=1)
    logits = model(input)
    log_probs = F.log_softmax(logits, dim=-1)
    u_values = torch.tensor([-1, 0, 1])
    with torch.no_grad():
        probs = log_probs.exp()
        index = torch.multinomial(probs, num_samples=1)
        u = u_values[index].squeeze(1)
        r = reward(x_0, x_ref, u)
    selected_log_probs = log_probs.gather(dim=1, index=index).squeeze(1)
    value = (r * selected_log_probs).mean()
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
    
