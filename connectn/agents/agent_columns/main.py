import numpy as np
from typing import Tuple
from connectn.utils import SavedState
from connectn.game import NO_PLAYER


class SavedColumn(SavedState):
    def __init__(self, col):
        self.col = col


def generate_move(
    board: np.ndarray, player: int, saved_state: SavedColumn
) -> Tuple[int, SavedColumn]:
    if saved_state is None:
        saved_state = SavedColumn(np.random.randint(board.shape[1], size=1))
    prev_col = saved_state.col
    cols = np.roll(np.arange(board.shape[1]), -prev_col)
    j = -1
    for j in cols:
        if board[-1, j] == NO_PLAYER:
            break
    saved_state.col = j
    return j, saved_state
