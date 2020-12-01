import pytest
import numpy as np
from connectn.game import PLAYER1, PLAYER2
from connectn.utils import SavedState


class TestState(SavedState):
    def __init__(self, test):
        self.test = test


def test_generate_move_process(empty_board: np.ndarray):
    from multiprocessing import Queue
    from connectn.tournament import (
        generate_move_process,
        GenMoveSuccess,
        GenMoveFailure,
        GenMoveArgs,
    )

    q = Queue()
    seed = None
    a = 6543

    init_state = TestState(9876)

    gma = GenMoveArgs(seed, empty_board, PLAYER1, init_state)
    q.put(gma)
    generate_move_process(lambda board, player, saved_state: (a, saved_state), q)
    ret: GenMoveSuccess = q.get()

    returned_state: TestState = ret.state
    action = ret.action

    assert action == a
    assert returned_state.test == init_state.test

    def fail_move(_board, _player, _saved_state):
        raise Exception("Test failure")

    gma = GenMoveArgs(seed, empty_board, PLAYER1, init_state)
    q.put(gma)
    generate_move_process(fail_move, q)
    ret: GenMoveFailure = q.get()
    error_msg = ret.error_msg
    assert error_msg.startswith("Exception('Test failure',)")
    assert error_msg.endswith('raise Exception("Test failure")\n')


def test_run_single_game():
    from connectn.tournament import run_single_game

    with pytest.raises(KeyError):
        run_single_game("doesnotexist", "agent_rows")

    gr = run_single_game("agent_rows", "agent_fail")
    assert gr.winner == PLAYER1
    assert gr.result_1.outcome == "WIN" and gr.result_1.name == "agent_rows"
    assert gr.result_2.outcome == "FAIL" and gr.result_2.name == "agent_fail"

    gr = run_single_game("agent_fail", "agent_rows")
    assert gr.winner == PLAYER2
    assert gr.result_1.outcome == "FAIL" and gr.result_1.name == "agent_fail"
    assert gr.result_2.outcome == "WIN" and gr.result_2.name == "agent_rows"

    seed = 0
    gr = run_single_game("agent_columns", "agent_rows", seed)
    assert gr.winner == PLAYER1
    assert gr.result_1.outcome == "WIN" and gr.result_2.outcome == "LOSS"
    assert gr.result_1.moves == [5, 5, 5, 5]
    assert gr.result_2.moves == [0, 1, 2]
    assert gr.result_1.name == "agent_columns" and gr.result_2.name == "agent_rows"


def test_run_tournament_cluster(monkeypatch):
    from connectn.tournament import run_tournament_cluster, GameResult, AgentResult
    import connectn.results as results
    import gridmap

    def mock_grid_map(fn, args_list, **kwargs):
        """  f, args_list, cleanup=True, mem_free="1G", name='gridmap_job',
             num_slots=1, temp_dir=DEFAULT_TEMP_DIR, white_list=None,
             queue=DEFAULT_QUEUE, quiet=True, local=False, max_processes=1,
             interpreting_shell=None, copy_env=True, add_env=None,
             completion_mail=False, require_cluster=False, par_env=DEFAULT_PAR_ENV"""
        assert callable(fn)
        assert "mem_free" in kwargs and kwargs["mem_free"] == "2G"
        assert "queue" in kwargs and kwargs["queue"] == "cognition-all.q"
        assert "require_cluster" in kwargs and kwargs["require_cluster"]
        assert "add_env" in kwargs
        add_env = kwargs["add_env"]
        assert "USE_MEM_FREE" in add_env
        assert "CREATE_PLOTS" in add_env
        for agent_1, agent_2 in args_list:
            yield GameResult(AgentResult(agent_1), AgentResult(agent_2))

    def mock_add_game(game_result):
        assert isinstance(game_result, GameResult)

    monkeypatch.setattr(gridmap, "grid_map", mock_grid_map)
    monkeypatch.setattr(results, "add_game", mock_add_game)
    run_tournament_cluster(
        [["group_a", "group_b"], ]
    )


def test_run_tournament_process(monkeypatch):
    from connectn import results, utils, tournament
    from connectn.tournament import run_tournament_process
    from connectn.users import agents
    from connectn.tournament import GameResult, AgentResult
    from multiprocessing import Queue, Event
    from functools import partial

    all_agents = agents()

    def mock_init(_agents):
        for agent in _agents:
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

    def mock_update_user_agent_code(_updated_agents, updated_agent_archives):
        for agent_name_archive_path in updated_agent_archives:
            assert agent_name_archive_path in _updated_agents
        return [agent_name for agent_name, _ in updated_agent_archives]

    def mock_run_single_game(_agent_names, agent_1, agent_2):
        assert agent_1 in _agent_names or agent_2 in _agent_names
        assert agent_1 != agent_2
        assert agent_1 in all_agents and agent_2 in all_agents
        return GameResult(AgentResult(agent_1), AgentResult(agent_2))

    sq = Queue()
    rq = Queue()
    ev = Event()
    updated_agents_a = [("agent_rows", "no file")]

    updated_agents_b = [("agent_columns", "no file"), ("agent_random", "no file")]
    for updated_agents in (updated_agents_a, updated_agents_b):
        agent_names = [agent_name for agent_name, _ in updated_agents]
        monkeypatch.setattr(
            tournament, "run_single_game", partial(mock_run_single_game, agent_names)
        )
        monkeypatch.setattr(
            utils,
            "update_user_agent_code",
            partial(mock_update_user_agent_code, updated_agents),
        )

        sq.put(updated_agents)
        ev.set()
        run_tournament_process(sq, rq, ev, play_all=False)
