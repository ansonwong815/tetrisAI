from typing import Tuple, Dict
import numpy as np
import cv2
from PIL import Image


class TetrisRenderer:
    def __init__(self, size: Tuple, block_size: int, colours: Dict):
        self.size = size
        self.block_size = block_size
        self.colours = colours
        cv2.destroyAllWindows()

    def render(self, state):
        # cv2.destroyAllWindows() #hv to have this?
        img = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        for h, row in enumerate(state):
            for w, block in enumerate(row):
                img[h, w] = self.colours[block]

        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")
        img = img.resize((self.size[1] * self.block_size, self.size[0] * self.block_size), Image.NEAREST)

        img = np.array(img)
        cv2.imshow("Tetris stuff", img)
        cv2.waitKey(1)
