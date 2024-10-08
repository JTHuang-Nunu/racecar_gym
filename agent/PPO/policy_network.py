import torch.nn as nn

from .distribution_module import DiagGaussian


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        # super().__init__()
        # self.main = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU()
        # )
        # self.dist = DiagGaussian(128, action_dim)

        # Alter: 128->256
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.dist = DiagGaussian(128, action_dim)

    def forward(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist.log_probs(action)

    def choose_action(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            return dist.mode()

        return dist.sample()

    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        return dist.log_probs(action), dist.entropy()
