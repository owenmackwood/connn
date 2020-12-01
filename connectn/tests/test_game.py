import numpy as np
import pytest
from connectn.game import CONNECT_N, PLAYER1, PLAYER2, GameStatus


def test_apply_player_action_success(empty_board: np.ndarray):
    from connectn.game import apply_player_action

    for col in np.arange(empty_board.shape[1], dtype=empty_board.dtype):
        for player in (PLAYER1, PLAYER2):
            b1 = apply_player_action(empty_board, col, player, copy=True)
            bd = np.argwhere(b1 - empty_board)
            assert bd.shape == (1, 2), "Board changed in more than one place"
            bd = bd[0, :]
            assert (
                bd[1] == col
            ), f"Requested column {col}, was not played. Played in {bd[1]} instead"
            assert b1[bd[0], bd[1]] == player, "Used wrong player identifier"
            assert empty_board[bd[0], col] == 0, "Played in a non-empty space"
            assert (
                bd[0] == 0 or empty_board[bd[0] - 1, col] > 0
            ), "Played above an empty space"


def test_apply_player_action_fail(full_board_p1: np.ndarray):
    from connectn.game import apply_player_action
    from connectn.tournament import AgentFailed

    for col in np.arange(full_board_p1.shape[1], dtype=full_board_p1.dtype):
        with pytest.raises(AgentFailed):
            apply_player_action(full_board_p1, col, PLAYER1)


def test_check_end_state_rows(empty_board):
    from connectn.game import check_end_state, other_player

    for i in range(CONNECT_N - 1):
        for row in range(empty_board.shape[0]):
            for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row, col:col + CONNECT_N - i] = player
                    b0[:row, col:col + CONNECT_N - i] = other_player(player)
                    if i == 0:
                        assert check_end_state(b0, player) == GameStatus.IS_WIN
                    else:
                        assert check_end_state(b0, player) == GameStatus.STILL_PLAYING


def test_check_end_state_cols(empty_board):
    from connectn.game import check_end_state, other_player

    for i in range(CONNECT_N - 1):
        for col in range(empty_board.shape[1]):
            for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row:row + CONNECT_N - i, col] = player
                    b0[:row, col] = other_player(player)
                    if i == 0:
                        assert check_end_state(b0, player) == GameStatus.IS_WIN
                    else:
                        assert check_end_state(b0, player) == GameStatus.STILL_PLAYING


@pytest.mark.xfail(
    reason=""" 
No longer works with modified check_end_state due to early termination
of row / column search if empty spots found.
"""
)
def test_check_end_state_straight(empty_board: np.ndarray):
    from connectn.game import check_end_state
    from itertools import product

    dim = len(empty_board.shape)
    for player in (PLAYER1, PLAYER2):
        for n in range(CONNECT_N):
            for d in range(dim):
                for i in range(empty_board.shape[d] - CONNECT_N + n + 1):
                    remaining_indices = list(
                        range(empty_board.shape[k]) for k in range(dim) if k != d
                    )
                    for jk in product(*remaining_indices):
                        b0 = empty_board.copy()
                        indices = jk[:d] + (slice(i, i + CONNECT_N - n),) + jk[d:]
                        b0[indices] = player
                        if n == 0:
                            assert check_end_state(b0, player) == GameStatus.IS_WIN
                        else:
                            assert check_end_state(b0, player) == GameStatus.STILL_PLAYING


def test_check_end_state_diagonal(empty_board: np.ndarray):
    from connectn.game import check_end_state

    for i in range(CONNECT_N):
        n_diagonal = np.diag(np.ones(CONNECT_N - i, dtype=empty_board.dtype))
        for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
            for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                    for player in (PLAYER1, PLAYER2):
                        b0 = empty_board.copy()
                        b0[row:row + CONNECT_N - i, col:col + CONNECT_N - i] = (
                            player * n_conn
                        )
                        if i == 0:
                            assert check_end_state(b0, player) == GameStatus.IS_WIN
                        else:
                            assert check_end_state(b0, player) == GameStatus.STILL_PLAYING


def test_connected_four_last_action(empty_board: np.ndarray):
    from connectn.game import connected_four, other_player

    for player in (PLAYER1, PLAYER2):
        for i in range(CONNECT_N):
            for row in range(empty_board.shape[0]):
                for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                    b0 = empty_board.copy()
                    b0[row, col:col + CONNECT_N - i] = player
                    for last_action in np.arange(col, col + CONNECT_N - i):
                        if i == 0:
                            assert connected_four(b0, player, last_action)
                        else:
                            assert not connected_four(b0, player, last_action)

            for col in np.arange(empty_board.shape[1]):
                for row in np.arange(empty_board.shape[0] - CONNECT_N + i + 1):
                    b0 = empty_board.copy()
                    b0[:row, col] = other_player(player)
                    b0[row:row + CONNECT_N - i, col] = player
                    if i == 0:
                        assert connected_four(b0, player, col)
                    else:
                        assert not connected_four(b0, player, col)

            n_diagonal = player * np.diag(
                np.ones(CONNECT_N - i, dtype=empty_board.dtype)
            )
            for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
                for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                    for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                        b0 = empty_board.copy()
                        b0[
                            row:row + CONNECT_N - i, col:col + CONNECT_N - i
                        ] = n_conn
                        for last_action in np.arange(col, col + CONNECT_N - i):
                            if i == 0:
                                assert connected_four(b0, player, last_action)
                            else:
                                assert not connected_four(b0, player, last_action)


def test_connected_four_oracle(empty_board: np.ndarray):
    from connectn.game import check_end_state, other_player, apply_player_action
    from connectn.game import PlayerAction, NO_PLAYER
    from connectn.game import connected_four_convolve, connected_four, connected_four_generators

    for game_n in range(1000):
        board = empty_board.copy()
        curr_player = PLAYER1
        end_state = check_end_state(board, curr_player)
        move_n = 0
        while end_state == GameStatus.STILL_PLAYING:
            move_n += 1
            curr_player = other_player(curr_player)
            moves = np.array(
                [
                    col
                    for col in np.arange(PlayerAction(board.shape[1]))
                    if board[PlayerAction(-1), col] == NO_PLAYER
                ]
            )
            move = np.random.choice(moves, 1)[0]
            apply_player_action(board, move, curr_player)
            end_state = check_end_state(board, curr_player)

            conn_4_a = connected_four_convolve(board, curr_player)
            conn_4_b = connected_four(board, curr_player, move, True)
            conn_4_b2 = connected_four(board, curr_player, move, False)
            conn_4_c = connected_four(board, curr_player, None, True)
            conn_4_c2 = connected_four(board, curr_player, None, False)
            conn_4_u = connected_four_generators(board, curr_player, move)
            assert conn_4_a == conn_4_b
            assert conn_4_b == conn_4_b2
            assert conn_4_b2 == conn_4_c
            assert conn_4_c == conn_4_c2
            assert conn_4_c2 == conn_4_u


def test_connected_four_convolve(empty_board: np.ndarray):
    from connectn.game import connected_four_convolve, other_player

    for i in range(CONNECT_N - 1):
        for row in range(empty_board.shape[0]):
            for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row, col:col + CONNECT_N - i] = player
                    b0[:row, col:col + CONNECT_N - i] = other_player(player)
                    if i == 0:
                        assert connected_four_convolve(b0, player)
                    else:
                        assert not connected_four_convolve(b0, player)

        for col in range(empty_board.shape[1]):
            for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                for player in (PLAYER1, PLAYER2):
                    b0 = empty_board.copy()
                    b0[row:row + CONNECT_N - i, col] = player
                    b0[:row, col] = other_player(player)
                    if i == 0:
                        assert connected_four_convolve(b0, player)
                    else:
                        assert not connected_four_convolve(b0, player)

    for i in range(CONNECT_N):
        n_diagonal = np.diag(np.ones(CONNECT_N - i, dtype=empty_board.dtype))
        for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
            for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                    for player in (PLAYER1, PLAYER2):
                        b0 = empty_board.copy()
                        b0[row:row + CONNECT_N - i, col:col + CONNECT_N - i] = (
                            player * n_conn
                        )
                        if i == 0:
                            assert connected_four_convolve(b0, player)
                        else:
                            assert not connected_four_convolve(b0, player)


def test_other_player():
    from connectn.game import other_player

    assert other_player(PLAYER1) == PLAYER2
    assert other_player(PLAYER2) == PLAYER1
