import numpy as np
from typing import Tuple
from connectn.utils import SavedState
from connectn.game import NO_PLAYER

def generate_move(board: np.ndarray, player: int, saved_state: SavedState) -> Tuple[int, SavedState]:
    cols = np.arange(board.shape[1])
    np.random.shuffle(cols)
    for col in cols:
        for row in range(board.shape[0]):
            if board[row, col] == NO_PLAYER:
                board[row, col] = player
                return col, saved_state
    return -1, saved_state
