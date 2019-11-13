import numpy as np
from connectn.game import apply_player_action, AgentFailed, check_end_state
from connectn.game import CONNECT_N, STILL_PLAYING, IS_WIN, PLAYER1, PLAYER2, btype
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
            apply_player_action(full_board_p1, col, PLAYER1)
        except AgentFailed:
            pass
        else:
            assert False, "Failed to throw exception when playing in full column."


def test_check_end_state_rows(empty_board):
    for i in range(CONNECT_N-1):
        for row in range(empty_board.shape[0]):
            for col in range(empty_board.shape[1]-CONNECT_N+i+1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row, col:col+CONNECT_N-i] = player
                    if i == 0:
                        assert check_end_state(b0, player) == IS_WIN
                    else:
                        assert check_end_state(b0, player) == STILL_PLAYING


def test_check_end_state_cols(empty_board):
    for i in range(CONNECT_N-1):
        for col in range(empty_board.shape[1]):
            for row in range(empty_board.shape[0]-CONNECT_N+i+1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row:row+CONNECT_N-i, col] = player
                    if i == 0:
                        assert check_end_state(b0, player) == IS_WIN
                    else:
                        assert check_end_state(b0, player) == STILL_PLAYING


def test_check_end_state_diagonal(empty_board):
    for i in range(CONNECT_N-1):
        n_diagonal = np.diag(PLAYER1*np.ones(CONNECT_N-i, dtype=btype))[:, ::-1]
        for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
            for row in range(empty_board.shape[0]-CONNECT_N+i+1):
                for col in range(empty_board.shape[1]-CONNECT_N+i+1):
                    for player in (PLAYER1, PLAYER2):
                        b0 = empty_board.copy()
                        b0[row:row+CONNECT_N-i, col:col+CONNECT_N-i] = n_conn
                        if i == 0:
                            assert check_end_state(b0, player) == IS_WIN
                        else:
                            assert check_end_state(b0, player) == STILL_PLAYING
