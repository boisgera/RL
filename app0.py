import itertools

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

"""
We consider a "mini-breakout" game where the paddle should catch the ball.
Everything takes places in a 3x2 grid. The x-coordinate is measured as 
-1 (left), 0 (center) or 1 (right); the paddle is initially at the lower
level, at the center and the ball pops at the upper level at a random
(uniform) position, and will drop vertically at next tick. The paddle
can decide to stay at its position, move left or move right for the next
tick to try and catch the ball.
"""

def sample_x_ref(num_samples=1):
    "Random values in {-1, 0, 1} (as flots)"
    return torch.randint(-1, 2, (num_samples,)).float()

def Model(hidden_sizes=()):
    "Input: target position, output: action policy (3 logits). hidden_sizes: size of each hidden layer, e.g. [] or [16] or [16, 16]."
    sizes = [1, *hidden_sizes, 3]
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

"""
We assume the padle is initially centered. Consequently, its location after
the control is applied is going to be equal to the control.
"""

def reward(x_ref, u):
    x = u  # since x = x0 + u and x0 = 0
    return (x == x_ref).float()

def mean_reward(model, num_samples=1_000):
    x_ref = sample_x_ref(num_samples)
    policy = model(x_ref.unsqueeze(1))
    u = sample_u(policy, num_samples).squeeze(1)
    r = reward(x_ref, u)
    return r.mean().item()

def mean_reward_grad(model, num_samples=1_000):
    """
    Estimate the gradient of the mean reward wrt model weights 
    using the REINFORCE log-derivative trick and sampling.
    The result is stored in `model.grad`.
    """
    n = num_samples
    x_ref = sample_x_ref(n)
    x_ref = x_ref.reshape((n, -1)) # batch of target positions
    logits = model(x_ref)
    log_probs = F.log_softmax(logits, dim=-1)
    u_values = torch.tensor([-1, 0, 1])
    with torch.no_grad():
        probs = log_probs.exp()
        # We sample the control policy once for each x_ref sample;
        # index stays 2d (n, 1) to match x_ref/log_probs for gather below.
        index = torch.multinomial(probs, num_samples=1)
        u = u_values[index]
        r = reward(x_ref, u)
    selected_log_probs = log_probs.gather(dim=1, index=index)
    value = (r * selected_log_probs).mean()
    model.zero_grad()
    value.backward()

def plot(model):
    with torch.inference_mode():
        x_ref = torch.linspace(-1.0, 1.0, 1000)
        logits = model(x_ref.reshape((-1, 1))) # batch dim. is index 0
    logits = logits.detach()
    p = logits.softmax(dim=-1)
    x_ref = x_ref.detach()

    fig, (ax_p, ax_logits) = plt.subplots(2, 1, sharex=True)

    ax_p.plot(x_ref, p[:, 0], label=r"$\mathbb{P}(u=-1)$")
    ax_p.plot(x_ref, p[:, 1], label=r"$\mathbb{P}(u=0)$")
    ax_p.plot(x_ref, p[:, 2], label=r"$\mathbb{P}(u=+1)$")
    ax_p.set_xlim(-1, 1)
    ax_p.set_ylim(0, 1)
    ax_p.set_ylabel(r"$\mathbb{P}$")
    ax_p.set_title(f"Mean reward: {mean_reward(model):.3f}")
    ax_p.grid(True)
    ax_p.legend()

    ax_logits.plot(x_ref, logits[:, 0], label="logit for u=-1")
    ax_logits.plot(x_ref, logits[:, 1], label="logit for u=0")
    ax_logits.plot(x_ref, logits[:, 2], label="logit for u=+1")
    ax_logits.set_xlabel("x_ref")
    ax_logits.set_ylabel("logit")
    ax_logits.grid(True)
    ax_logits.legend()

def train(model, steps=5_000):
    optimizer = optim.Adam(model.parameters(), maximize=True)
    print(f"Step 00 -- reward: {mean_reward(model)}")
    for i in range(steps):
        mean_reward_grad(model)
        optimizer.step()
        print(f"Step {i+1:2} -- reward: {mean_reward(model)}")


if __name__ == "__main__":
    model = Model([])
    train(model, steps=10_000)
    plot(model)
    plt.show()
    
