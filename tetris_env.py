import gym
import numpy as np

from game import TetrisGame


class TetrisEnv(gym.Env):
    def __init__(self, size, renderer, render=False):
        self.game = TetrisGame(size, renderer, render)
        self.size = size
        self.height = 0

    def _get_observation(self):
        observation = []
        for i in range(40):
            new_state, lines_cleared, done = self.game.simulate(i)
            space = 0
            num_holes = 0
            testnum = 0

            layer_heights = (22 - new_state.argmax(axis=0)) * new_state.max(axis=0)
            for j in range(self.size[1]):
                for b in range(int(layer_heights[j])):
                    if new_state[self.size[0] - b + 1][j] == 0:
                        if new_state[self.size[0] - b + 0][j] == 1:
                            num_holes += 1
                        space += 1

            bumpiness = 0
            last_height = layer_heights[0]
            for height in layer_heights:
                bumpiness += abs(height - last_height)
                last_height = height
            total_height = sum(layer_heights)
            max_height = max(layer_heights)
            min_height = min(layer_heights)
            observation.append([lines_cleared, num_holes, bumpiness, total_height, max_height, min_height, space])
        observation = np.array(observation)
        return observation

    def step(self, action):
        lines_cleared, done = self.game.move(action)
        layer_heights = (20 - self.game.board.argmax(axis=0)) * self.game.board.max(axis=0)
        max_height = max(layer_heights)
        reward = 1 - ((max_height - self.height) ** 2) + lines_cleared * 20 + done * -50
        self.height = max_height

        return self._get_observation(), reward, done, lines_cleared

    def reset(self):
        self.game.reset()
        return self._get_observation()

    def render(self, mode="human"):
        self.game.render()
