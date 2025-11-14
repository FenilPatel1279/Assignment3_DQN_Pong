import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from model import DQNCNN
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, action_size, gamma=0.95, lr=1e-4):
        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.memory = ReplayBuffer()

        self.policy_net = DQNCNN(action_size).to(device)
        self.target_net = DQNCNN(action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_vals = self.policy_net(state)
        return torch.argmax(q_vals).item()

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        dones = torch.tensor(dones, device=device)

        q_vals = self.policy_net(states)
        q = q_vals[range(batch_size), actions]

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
