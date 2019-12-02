import numpy as np
import pytest
from connectn.game import CONNECT_N, STILL_PLAYING, IS_WIN, PLAYER1, PLAYER2


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
    from connectn.game import apply_player_action, AgentFailed

    for col in np.arange(full_board_p1.shape[1], dtype=full_board_p1.dtype):
        try:
            apply_player_action(full_board_p1, col, PLAYER1)
        except AgentFailed:
            pass
        else:
            assert False, "Failed to throw exception when playing in full column."


def test_check_end_state_straight(empty_board: np.ndarray):
    from connectn.game import check_end_state
    from itertools import product

    dim = len(empty_board.shape)
    for player in (PLAYER1, PLAYER2):
        for l in range(CONNECT_N):
            for d in range(dim):
                for i in range(empty_board.shape[d] - CONNECT_N + l + 1):
                    remaining_indices = list(
                        range(empty_board.shape[k]) for k in range(dim) if k != d
                    )
                    for jk in product(*remaining_indices):
                        b0 = empty_board.copy()
                        indices = jk[:d] + (slice(i, i + CONNECT_N - l),) + jk[d:]
                        b0[indices] = player
                        if l == 0:
                            assert check_end_state(b0, player) == IS_WIN
                        else:
                            assert check_end_state(b0, player) == STILL_PLAYING


def test_check_end_state_diagonal(empty_board: np.ndarray):
    from connectn.game import check_end_state

    for i in range(CONNECT_N):
        n_diagonal = np.diag(np.ones(CONNECT_N - i, dtype=empty_board.dtype))
        for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
            for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                    for player in (PLAYER1, PLAYER2):
                        b0 = empty_board.copy()
                        b0[row : row + CONNECT_N - i, col : col + CONNECT_N - i] = (
                            player * n_conn
                        )
                        if i == 0:
                            assert check_end_state(b0, player) == IS_WIN
                        else:
                            assert check_end_state(b0, player) == STILL_PLAYING


def test_connected_four_last_action(empty_board: np.ndarray):
    from connectn.game import connected_four, other_player

    for player in (PLAYER1, PLAYER2):
        for i in range(CONNECT_N):
            for row in range(empty_board.shape[0]):
                for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                    b0 = empty_board.copy()
                    b0[row, col : col + CONNECT_N - i] = player
                    for last_action in np.arange(empty_board.shape[1]):
                        if i == 0 and col <= last_action < col + CONNECT_N:
                            assert connected_four(b0, player, last_action)
                        else:
                            assert not connected_four(b0, player, last_action)

            for col in range(empty_board.shape[1]):
                for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                    b0 = empty_board.copy()
                    b0[:row, col] = other_player(player)
                    b0[row : row + CONNECT_N - i, col] = player
                    for last_action in np.arange(empty_board.shape[1]):
                        if i == 0 and last_action == col:
                            assert connected_four(b0, player, last_action)
                        else:
                            assert not connected_four(b0, player, last_action)

            n_diagonal = player * np.diag(
                np.ones(CONNECT_N - i, dtype=empty_board.dtype)
            )
            for n_conn in (n_diagonal, n_diagonal[:, ::-1]):
                for row in range(empty_board.shape[0] - CONNECT_N + i + 1):
                    for col in range(empty_board.shape[1] - CONNECT_N + i + 1):
                        b0 = empty_board.copy()
                        b0[
                            row : row + CONNECT_N - i, col : col + CONNECT_N - i
                        ] = n_conn
                        for last_action in np.arange(empty_board.shape[1]):
                            if i == 0 and col <= last_action < col + CONNECT_N:
                                assert connected_four(b0, player, last_action)
                            else:
                                assert not connected_four(b0, player, last_action)
                        # if i == 0:
                        #     assert connected_four(b0, player)
                        # else:
                        #     assert not connected_four(b0, player)


def test_other_player():
    from connectn.game import other_player

    assert other_player(PLAYER1) == PLAYER2
    assert other_player(PLAYER2) == PLAYER1


def test_generate_move_process(empty_board: np.ndarray):
    from multiprocessing import Queue
    from connectn.utils import SavedState
    from connectn.game import generate_move_process, STATUS_SUCCESS, STATUS_FAILURE

    q = Queue()
    seed = None
    a = 6543
    init_state = SavedState()
    init_state.test = 9876
    q.put((seed, empty_board, PLAYER1, init_state))
    generate_move_process(lambda board, player, saved_state: (a, saved_state), q)
    ret = q.get()
    assert len(ret) == 5
    status, size, move_time, action, returned_state = ret
    assert status == STATUS_SUCCESS
    assert action == a
    assert returned_state.test == init_state.test

    def fail_move(board, player, saved_state):
        raise Exception("Test failure")

    q.put((seed, empty_board, PLAYER1, init_state))
    generate_move_process(fail_move, q)
    ret = q.get()
    assert len(ret) == 2
    status, error_msg = ret
    assert status == STATUS_FAILURE
    assert error_msg.startswith("Exception('Test failure')")
    assert error_msg.endswith('raise Exception("Test failure")\n')


def test_run_game_local():
    from connectn.game import run_game_local

    try:
        run_game_local("doesnotexist", "agent_rows")
    except KeyError:
        pass
    else:
        assert False

    gr = run_game_local("agent_rows", "agent_fail")
    assert gr.winner == PLAYER1
    assert gr.result_1.outcome == "WIN" and gr.result_1.name == "agent_rows"
    assert gr.result_2.outcome == "FAIL" and gr.result_2.name == "agent_fail"

    gr = run_game_local("agent_fail", "agent_rows")
    assert gr.winner == PLAYER2
    assert gr.result_1.outcome == "FAIL" and gr.result_1.name == "agent_fail"
    assert gr.result_2.outcome == "WIN" and gr.result_2.name == "agent_rows"

    seed = 0
    gr = run_game_local("agent_columns", "agent_rows", seed)
    assert gr.winner == PLAYER1
    assert gr.result_1.outcome == "WIN" and gr.result_2.outcome == "LOSS"
    assert gr.result_1.moves == 4 * [4] and gr.result_2.moves == [0, 1, 2]
    assert gr.result_1.name == "agent_columns" and gr.result_2.name == "agent_rows"


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_run_game_cluster():
    from connectn.game import run_game_cluster

    run_game_cluster("", "")


def test_run_games(monkeypatch):
    from connectn import results, utils, game
    from connectn.game import run_games
    from connectn.users import agents
    from connectn.game import GameResult, AgentResult
    from multiprocessing import Queue
    from functools import partial

    all_agents = agents()

    def mock_init(agents):
        for agent in agents:
            assert isinstance(agent, str)
            assert agent in all_agents

    def mock_add_agent(agent):
        assert isinstance(agent, str)
        assert agent in all_agents

    def mock_add_game(gr):
        assert isinstance(gr, GameResult)

    monkeypatch.setattr(results, "initialize", mock_init)
    monkeypatch.setattr(results, "add_agent", mock_add_agent)
    monkeypatch.setattr(results, "add_game", mock_add_game)

    def mock_update_user_agent_code(updated_agents, updated_agent_archives):
        for agent_name_archive_path in updated_agent_archives:
            assert agent_name_archive_path in updated_agents
        return [agent_name for agent_name, _ in updated_agent_archives]

    def mock_run_game(agent_names, agent_1, agent_2):
        assert agent_1 in agent_names or agent_2 in agent_names
        assert agent_1 != agent_2
        assert agent_1 in all_agents and agent_2 in all_agents
        return GameResult(AgentResult(agent_1, []), AgentResult(agent_2, []))

    q = Queue()
    updated_agents_a = [("agent_rows", "no file")]

    updated_agents_b = [("agent_columns", "no file"), ("agent_random", "no file")]
    for updated_agents in (updated_agents_a, updated_agents_b):
        agent_names = [agent_name for agent_name, _ in updated_agents]
        monkeypatch.setattr(game, "run_game", partial(mock_run_game, agent_names))
        monkeypatch.setattr(
            utils,
            "update_user_agent_code",
            partial(mock_update_user_agent_code, updated_agents),
        )

        q.put(updated_agents)
        q.put("SHUTDOWN")
        run_games(q, play_all=False)
