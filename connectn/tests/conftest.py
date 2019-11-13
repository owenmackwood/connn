import pytest
import numpy as np
from connectn.game import BOARD_SHAPE, btype, PLAYER1, PLAYER2

@pytest.fixture
def empty_board():
    return np.zeros(BOARD_SHAPE, dtype=btype)


@pytest.fixture
def full_board_p1():
    return PLAYER1 * np.ones(BOARD_SHAPE, dtype=btype)


@pytest.fixture
def full_board_p2():
    return PLAYER2 * np.ones(BOARD_SHAPE, dtype=btype)

