import random
from typing import Tuple

import numpy as np

from renderer import TetrisRenderer


class TetrisGame:
    def __init__(self, size: Tuple, renderer: TetrisRenderer, render=False):
        self.size = size
        self.renderer = renderer
        self.board = np.zeros((self.size[0] + 2, self.size[1]))  # invisible rows
        self.is_render = render
        self.position = None
        self.rotation = 0
        self.piece = None
        self.options = [0, 1, 2, 3, 4, 5];
        self.piece_map = [
            [
                [[0, 0], [0, 1], [0, -1], [0, -2]],
                [[0, 0], [-1, 0], [-2, 0], [1, 0]],
                [[0, 0], [0, 1], [0, -1], [0, -2]],
                [[0, 0], [-1, 0], [-2, 0], [1, 0]],
            ],

            [
                [[0, 0], [0, 1], [0, -1], [1, 1]],
                [[0, 0], [1, 0], [-1, 0], [1, -1]],
                [[0, 0], [0, 1], [0, -1], [-1, -1]],
                [[0, 0], [1, 0], [-1, 0], [-1, 1]]
            ],

            [
                [[0, 0], [0, 1], [0, -1], [1, -1]],
                [[0, 0], [1, 0], [-1, 0], [-1, -1]],
                [[0, 0], [0, 1], [0, -1], [-1, 1]],
                [[0, 0], [1, 0], [-1, 0], [1, 1]]
            ],

            [
                [[0, 0], [0, 1], [1, 0], [1, -1]],
                [[0, 0], [-1, 0], [0, 1], [1, 1]],
                [[0, 0], [0, 1], [1, 0], [1, -1]],
                [[0, 0], [-1, 0], [0, 1], [1, 1]],
            ],

            [
                [[0, 0], [0, -1], [1, 0], [1, 1]],
                [[0, 0], [1, 0], [0, 1], [-1, 1]],
                [[0, 0], [0, -1], [1, 0], [1, 1]],
                [[0, 0], [1, 0], [0, 1], [-1, 1]],
            ],

            [
                [[0, 0], [0, -1], [-1, 0], [-1, -1]],
                [[0, 0], [0, -1], [-1, 0], [-1, -1]],
                [[0, 0], [0, -1], [-1, 0], [-1, -1]],
                [[0, 0], [0, -1], [-1, 0], [-1, -1]],
            ],

            [
                [[0, 0], [0, 1], [-1, 0], [1, 0]],
                [[0, 0], [0, 1], [0, -1], [-1, 0]],
                [[0, 0], [0, -1], [-1, 0], [1, 0]],
                [[0, 0], [0, -1], [0, 1], [-1, 0]]
            ]
        ]

        self._generate_new_piece()
        """
        Piece number
        0 - I-shape
        1 - 7-shape
        2 - L-shape
        3 - S shape
        4 - Z shape
        5 - square shape
        6 - T shape
        """
        """
        Board num
        0 - empty
        1 - filled
        """
    def _generate_new_piece(self):
        self.rotation = 0
        opt = random.randint(0,len(self.options)-1)
        self.piece = self.options[opt]
        self.options.pop(int(opt))
        self.position = (2, 4) if self.piece != 0 else (2,5)
        if len(self.options) == 0:
            self.options = [0, 1, 2, 3, 4, 5]
        lines_cleared = 0
        # clear lines
        while np.max(np.sum(self.board, axis=1)) == self.size[1]:
            # line is filled
            i = np.argmax(np.sum(self.board, axis=1))
            self.board = np.concatenate([np.zeros((1, self.size[1])), self.board[:i], self.board[i + 1:]])
            lines_cleared += 1

        if self._check_collision(self.position, self.rotation):
            # game finish
            return lines_cleared, True
        else:
            return lines_cleared, False

    def step(self, action):
        """
        action
        0 - no-op
        1 - left
        2 - right
        3 - down
        4 - clockwise rotation
        5 - counter-clockwise rotation
        """
        generated_piece, lines_cleared, done = False, 0, False
        temp_pos = self.position
        temp_rotation = self.rotation
        if action == 1:
            temp_pos = (temp_pos[0], temp_pos[1] - 1)
        if action == 2:
            temp_pos = (temp_pos[0], temp_pos[1] + 1)
        if action == 3:
            temp_pos = (temp_pos[0] + 1, temp_pos[1])
        if action == 4:
            temp_rotation = self.rotation + 1 if self.rotation < 3 else 0
        if action == 5:
            temp_rotation = self.rotation - 1 if self.rotation > 0 else 3

        if self._check_collision(temp_pos, temp_rotation):
            if action == 3:
                for pos in self.piece_map[self.piece][self.rotation]:
                    block_h = pos[0] + self.position[0]
                    block_w = pos[1] + self.position[1]
                    self.board[block_h, block_w] = 1

                lines_cleared, done = self._generate_new_piece()
                generated_piece = True
        # shd i gen new piece whenever confirm or wait one frame
        else:
            self.position = temp_pos
            self.rotation = temp_rotation

        if self.is_render:
            self.render()

        return generated_piece, lines_cleared, done

    def _in_range(self, pos):
        return not ((pos[0] >= self.size[0] + 2) or (pos[0] < 0) or (pos[1] >= self.size[1]) or (pos[1] < 0))

    def _check_collision(self, position, rotation):
        for pos in self.piece_map[self.piece][rotation]:
            block_h = pos[0] + position[0]
            block_w = pos[1] + position[1]
            if not self._in_range((block_h, block_w)):
                return True
            if self.board[block_h, block_w] != 0:
                return True

        return False

    def move(self, action):
        position = action % 10
        rotation = int(action / 10)
        delta_position = position - self.position[1]
        delta_rotation = rotation - self.rotation
        rotation_map = {
            -3: 1,
            -2: -2,
            -1: -1,
            0: 0,
            1: 1,
            2: 2,
            3: -1,
        }
        delta_rotation = rotation_map[delta_rotation]
        # change rotation
        if delta_rotation > 0:
            for i in range(abs(delta_rotation)):
                self.step(4)
        elif delta_rotation < 0:
            for i in range(abs(delta_rotation)):
                self.step(5)

        # change position
        if delta_position > 0:
            for i in range(delta_position):
                self.step(2)
        elif delta_position < 0:
            for i in range(abs(delta_position)):
                self.step(1)

        generated_piece, lines_cleared, done = False, 0, False
        while not generated_piece:
            generated_piece, lines_cleared, done = self.step(3)

        return lines_cleared, done

    def render(self):
        state = self.board[2:].copy()
        for pos in self.piece_map[self.piece][self.rotation]:
            block_h = pos[0] + self.position[0]
            block_w = pos[1] + self.position[1]
            if block_h >= 2:
                state[block_h - 2, block_w] = 2

        self.renderer.render(state)

    def reset(self):
        self.board = np.zeros((self.size[0] + 2, self.size[1]))
        self._generate_new_piece()

    def simulate(self, action):
        render_copy = self.is_render
        self.is_render = False

        data_copy = (self.board.copy(), self.piece, self.position, self.rotation)

        lines_cleared, done = self.move(action)
        new_state = self.board

        self.board, self.piece, self.position, self.rotation = data_copy

        self.is_render = render_copy

        return new_state, lines_cleared, done
