import numpy as np
from connectn.game import apply_player_action, AgentFailed, check_end_state
from connectn.game import IS_DRAW, PLAYER1, PLAYER2, btype
# from numpy.testing import assert_array_equal, assert_


def test_apply_player_action_success(empty_board):
    for col in np.arange(empty_board.shape[1], dtype=btype):
        for player in (PLAYER1, PLAYER2):
            b1 = apply_player_action(empty_board, col, player, copy=True)
            bd = np.argwhere(b1 - empty_board)
            assert bd.shape == (1, 2), "Board changed in more than one place"
            bd = bd[0, :]
            assert bd[1] == col, f"Requested column {col}, was not played. Played in {bd[1]} instead"
            assert b1[bd[0], bd[1]] == player, "Used wrong player identifier"
            assert empty_board[bd[0], col] == 0, "Played in a non-empty space"
            assert bd[0] == 0 or empty_board[bd[0] - 1, col] > 0, "Played above an empty space"


def test_apply_player_action_fail(full_board_p1):
    for col in np.arange(full_board_p1.shape[1], dtype=btype):
        try:
            apply_player_action(full_board_p1, col, btype(1))
        except AgentFailed:
            pass
        else:
            assert False, "Failed to throw exception when playing in full column."
