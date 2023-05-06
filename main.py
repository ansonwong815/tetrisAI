import time

import gym
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple, deque
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from renderer import TetrisRenderer
from tetris_env import TetrisEnv

Transition = namedtuple("Transition", ("state", "action", "reward", "done", "prev_state"))
device = torch.device("mps")


class ReplayMemory:
    def __init__(self, size, batch_size):
        self.size = size
        self.data = deque(maxlen=size)
        self.batch_size = batch_size

    def can_get_batch(self):
        return len(self.data) >= self.batch_size * 5

    def get_batch(self):
        batch = random.sample(self.data, self.batch_size)
        states = [i[0] for i in batch]
        actions = [i[1] for i in batch]
        rewards = [i[2] for i in batch]
        dones = [i[3] for i in batch]
        prev_states = [i[4] for i in batch]

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device).reshape(self.batch_size * 40, 7)
        actions = torch.tensor(actions).to(device).long().reshape(self.batch_size, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(np.invert(dones)).to(device).long()
        prev_states = torch.tensor(np.array(prev_states), dtype=torch.float32).to(device).reshape(self.batch_size * 40,
                                                                                                  7)
        return (states, actions, rewards, dones, prev_states)

    def store_batch(self, batch):
        self.data.append(batch)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)

        )

    def forward(self, x):
        return self.network(x)


batch_size = 256
replay_memory = ReplayMemory(50000, batch_size)
renderer = TetrisRenderer((20, 10), block_size=40, colours={0: (84, 84, 84), 1: (75, 179, 70), 2: (0, 162, 173)})
env = TetrisEnv((20, 10), renderer, render=False)

network = Network().to(device)
n_dict, o_dict = torch.load("models/3600_tetris.pth", map_location=torch.device('mps'))
network.load_state_dict(n_dict)
optim = torch.optim.Adam(network.parameters(), lr=1e-4)
optim.load_state_dict(o_dict)

target_network = Network().to(device)
target_network.load_state_dict(n_dict)

loss_fn = nn.MSELoss()
epsilon = 0.01
gamma = 0.99
tau = 0.005
writer = SummaryWriter()

# run loop
n = 0

for epoch in range(1, 2001):
    epsilon *= 0.998
    state = env.reset()
    done = False
    episodic_loss = 0
    episodic_reward = 0
    length = 0
    total_lines_cleared = 0
    while not done:
        length += 1
        with torch.no_grad():
            if random.random() < epsilon:
                action = random.randint(0, 39)
            else:
                state_t = torch.tensor(state, dtype=torch.float32).to(device)
                output = network(state_t)
                action = torch.argmax(output).item()

        prev_state = state
        state, reward, done, lines_cleared = env.step(action)
        env.render()
        total_lines_cleared += lines_cleared
        replay_memory.store_batch(Transition(state, action, reward, done, prev_state))
        episodic_reward += reward
        # train
        if replay_memory.can_get_batch():
            states, actions, rewards, dones, prev_states = replay_memory.get_batch()

            with torch.no_grad():
                q_label = target_network(states).reshape(batch_size, 40)
                q_label = torch.max(q_label, dim=1)[0] * dones
                q_label = rewards + gamma * q_label
                q_label = q_label.reshape(batch_size, 1)

            output = network(prev_states).reshape(batch_size, 40)
            q_pred = output.gather(1, actions)
            loss = loss_fn(q_pred, q_label)
            episodic_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1)

            optim.step()

            optim.zero_grad()
            network.zero_grad()

        # soft update target network
        for target_param, local_param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    writer.add_scalar("Epoch/reward", episodic_reward, epoch)
    writer.add_scalar("Epoch/loss", episodic_loss / length, epoch)
    writer.add_scalar("Epoch/epsilon", epsilon, epoch)
    writer.add_scalar("Epoch/lines cleared", total_lines_cleared, epoch)
    writer.add_scalar("Epoch/length", length, epoch)
    print(
        f"Episode {epoch} -- reward {episodic_reward} -- loss {episodic_loss / length} -- lines cleared {total_lines_cleared} -- length {length}")

    # test
    if epoch % 50 == 0:
        torch.save((network.state_dict(), optim.state_dict()), f"models/{epoch}_tetris.pth")
        state = env.reset()
        done = False
        episodic_loss = 0
        episodic_reward = 0
        testlength = 0
        while not done:
            with torch.no_grad():
                testlength += 1
                state_t = torch.tensor(state, dtype=torch.float32).to(device)
                output = network(state_t)
                action = torch.argmax(output).item()

            prev_state = state
            state, reward, done, lines_cleared = env.step(action)
            total_lines_cleared += lines_cleared
            env.render()
            episodic_reward += reward
            print(
                f"\rTest  {epoch}   -- reward {episodic_reward} -- length {testlength} -- line cleared {total_lines_cleared}",
                end="")
        print("")
        writer.add_scalar("Epoch/test reward", episodic_reward, epoch)
        writer.add_scalar("Epoch/test lines cleared", total_lines_cleared, epoch)

epoch = 0
while True:
    state = env.reset()
    epoch += 1
    done = False
    total_lines_cleared = 0
    episodic_loss = 0
    episodic_reward = 0
    length = 0
    while not done:
        with torch.no_grad():
            length += 1
            state_t = torch.tensor(state, dtype=torch.float32).to(device)
            output = network(state_t)
            action = torch.argmax(output).item()

        prev_state = state
        state, reward, done, lines_cleared = env.step(action)
        total_lines_cleared += lines_cleared
        env.render()
        episodic_reward += reward
        print(f"\rTest {epoch}  -- reward {episodic_reward} -- length {length} -- line cleared {total_lines_cleared}",
              end="")
    print("")
