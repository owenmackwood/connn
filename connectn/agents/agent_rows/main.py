import numpy as np
from typing import Tuple
from connectn.utils import SavedState
from connectn.game import EMPTY

class SavedRows(SavedState):
    def __init__(self, col: int):
        self.col = col

def generate_move(board: np.ndarray, player: int, saved_state: SavedRows) -> Tuple[int, SavedRows]:
    if saved_state is None:
        saved_state = SavedRows(board.shape[1] - 1)
    prev_col = saved_state.col
    cols = np.roll(np.arange(board.shape[1]), -(prev_col+1))
    for j in cols:
        for i in range(board.shape[0]):
            if board[i, j] == EMPTY:
                saved_state.col = j
                return j, saved_state

    return -1, saved_state

