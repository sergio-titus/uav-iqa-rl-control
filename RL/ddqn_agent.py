import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(
        self,
        state_dim,
        n_actions,
        gamma=0.99,
        lr=1e-4,
        buffer_size=200_000,
        batch_size=64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Q-networks
        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer (this is what train_ddqn.py expects)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Target update frequency (in gradient steps)
        self.target_update_freq = 1000
        self.learn_step = 0

    def act(self, state, epsilon: float):
        """Epsilon-greedy action."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        action = int(q_values.argmax(dim=1).item())
        return action

    def update(self):
        """One DDQN gradient step. Returns loss (float) or None."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones.astype(np.float32)).to(self.device)

        # Current Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # DDQN: use policy net to choose a*, target net to evaluate it
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())
