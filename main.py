import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import torch
from torch import nn
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

def Model(hidden_size=16):
    hidden_size = 16
    model = nn.Sequential(
        nn.Linear(1, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 3)
    )
    return model

def sample_u(policy, num_samples=1):
    """
    Random values of the control in {-1, 0, 1}

    The policy is the law of the control, as a tensor of 3 unnormalized log probas
    """
    u_values = torch.tensor([-1, 0, 1])
    probas = F.softmax(policy, dim=0)
    i = torch.multinomial(probas, num_samples=1)    
    u = u_values[i]
    return u

"""
We assume the padle is initially centered. Consequently, its location after
the control is applied is going to be equal to the control.
"""

def mean_reward(x_ref, u):
    x = u
    return (x == x_ref).float().mean()

def sample_reward(model, num_samples=1000):
    x_ref = sample_x_ref(num_samples)
    model = Model()
    policy = model(x_ref.unsqueeze(1))
    u = sample_u(policy, num_samples).squeeze(1)
    r = mean_reward(x_ref, u)
    return r.float().mean().item()

def plot(model):
    with torch.inference_mode():
        x_ref = torch.linspace(-1.0, 1.0, 1000)
        u = model(x_ref.reshape((-1, 1))) # batch dim. is index 0
        u = u.detach()
        x_ref = x_ref.detach()
        plt.plot(x_ref, u[:, 0], label="ulogp for u=-1")
        plt.plot(x_ref, u[:, 1], label="ulogp for u=0")
        plt.plot(x_ref, u[:, 2], label="ulogp for u=+1")
        ax = plt.gca()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        plt.legend()

if __name__ == "__main__":
    model = Model()
    plot(model); plt.show()
    print(f"mean reward: {sample_reward(model)}")
