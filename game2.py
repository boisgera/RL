# Third-Party Libraries
import torch
import pyxel

import app2

CELL = 24          # pixel size of one grid cell
STEP_FRAMES = 20    # animation frames per simulation tick
PAUSE_FRAMES = 30   # frames to hold the final result before the next round


def trajectory(model):
    "Sample a round and play it with the max-proba policy; return the paddle path and the target column."
    with torch.no_grad():
        i = torch.randint(0, app2.WIDTH, (1,)).item()
        j = torch.randint(0, app2.WIDTH, (1,)).item()
        x_ref = torch.tensor([2.0 * j / (app2.WIDTH - 1) - 1.0])
        x_t = torch.tensor([2.0 * i / (app2.WIDTH - 1) - 1.0])
        path = [i]
        for _ in range(app2.HEIGHT - 1):
            state = torch.stack((x_t, x_ref), dim=1)
            logits = model(state)
            u = torch.tensor([-1, 0, 1])[logits.argmax(dim=-1)]
            x_t = app2.step(x_t, u)
            path.append(round((x_t.item() + 1.0) / 2.0 * (app2.WIDTH - 1)))
    return path, j


class Game:
    def __init__(self, model):
        self.model = model
        pyxel.init(app2.WIDTH * CELL, app2.HEIGHT * CELL + 16, title="Catch the ball")
        self.new_round()
        pyxel.run(self.update, self.draw)

    def new_round(self):
        self.path, self.target = trajectory(self.model)
        self.frame0 = pyxel.frame_count

    def update(self):
        elapsed = pyxel.frame_count - self.frame0
        if elapsed >= STEP_FRAMES * (app2.HEIGHT - 1) + PAUSE_FRAMES:
            self.new_round()

    def draw(self):
        pyxel.cls(0)
        elapsed = pyxel.frame_count - self.frame0
        tick = min(elapsed // STEP_FRAMES, app2.HEIGHT - 1)
        pyxel.circ(self.target * CELL + CELL // 2, tick * CELL + CELL // 2, 5, 8)
        i = self.path[tick]
        hit = tick == app2.HEIGHT - 1 and i == self.target
        pyxel.rect(i * CELL, (app2.HEIGHT - 1) * CELL, CELL, 5, 11 if hit else 7)
        if tick == app2.HEIGHT - 1:
            pyxel.text(4, app2.HEIGHT * CELL + 4, "CAUGHT!" if hit else "MISSED", 11 if hit else 8)


if __name__ == "__main__":
    model = app2.Model([16])
    app2.train(model)
    Game(model)
