import time
import multiprocessing as mp
import numpy as np
import traceback
import logging
from typing import List, Union, Optional
from connectn.utils import GenMove, mib, SavedState
from connectn.users import import_agents, agents
from connectn.utils import MOVE_TIME_MAX, STATE_MEMORY_MAX, ON_CLUSTER
from connectn import IS_DEBUGGING
import numba as nb

logger = logging.getLogger(__name__)

CONNECT_N = np.int8(4)

PlayerAction = np.int8

BoardValue = np.int8
IS_WIN = BoardValue(1)
STILL_PLAYING = BoardValue(0)
IS_DRAW = BoardValue(-1)

BoardPiece = np.int8
NO_PLAYER = BoardPiece(0)
PLAYER1 = BoardPiece(1)
PLAYER2 = BoardPiece(2)


class AgentFailed(Exception):
    pass


class AgentResult:
    def __init__(self, agent_name: str):
        self.name: str = agent_name
        self.moves: List[PlayerAction] = []
        self.move_times: List[float] = []
        self.state_size: List[int] = []
        self.seeds: List[int] = []
        self.stdout: List[str] = []
        self.stderr: List[str] = []
        self.outcome: str = "NONE"


class GameResult:
    def __init__(self, agent_r1: AgentResult, agent_r2: AgentResult):
        self.result_1: AgentResult = agent_r1
        self.result_2: AgentResult = agent_r2
        self.winner: BoardPiece = NO_PLAYER
        self.time_sec: float = time.time()
        self.time_str: str = time.ctime()


class GenMoveArgs:
    def __init__(
        self,
        seed: Union[int, None],
        board: np.ndarray,
        player: BoardPiece,
        state: Optional[SavedState],
    ):
        self.seed = seed
        self.board = board
        self.player = player
        self.state = state


class GenMoveResult:
    def __init__(self, stdout: str, stderr: str):
        self.stdout = stdout
        self.stderr = stderr


class GenMoveSuccess(GenMoveResult):
    def __init__(
        self, stdout: str, stderr: str, move_time: float, action: int, state: SavedState
    ):
        super().__init__(stdout, stderr)
        self.move_time = move_time
        self.action = action
        self.state = state


class GenMoveFailure(GenMoveResult):
    def __init__(self, stdout: str, stderr: str, error_msg: str):
        super().__init__(stdout, stderr)
        self.error_msg = error_msg


def run_games(q: mp.Queue, play_all: bool = True):
    from itertools import product
    from queue import Empty as EmptyQueue
    from connectn.utils import update_user_agent_code
    import connectn.results as results

    repetitions = 1
    run_all_after = 60 * 60  # Run all-against-all every 60 minutes

    agent_modules = import_agents({})
    results.initialize(agent_modules.keys())
    while True:
        if play_all:
            play_all = False
            updated_agents = list(agents())
            logger.info(f"Just started, running all-against-all once.")
        else:
            # Check the message queue for updated agents
            logger.info("Game-playing process entering queue to wait for new agents")
            try:
                q_data = q.get(block=True, timeout=run_all_after)
            except EmptyQueue:
                updated_agents = list(agents())
                logger.info(
                    "Timed out waiting for new agents, running all-against-all."
                )
            else:
                if isinstance(q_data, str) and q_data == "SHUTDOWN":
                    return
                else:
                    updated_agents = update_user_agent_code(q_data)
                    logger.info(
                        f'Received {len(updated_agents)} updated agents for game-play: {" ".join(updated_agents)}'
                    )

        agent_modules = import_agents(agent_modules)
        agent_modules.pop("agent_fail", None)

        to_play = repetitions * [
            list(g)  # grid_map wants args as a list
            for g in product(agent_modules.keys(), agent_modules.keys())
            if g[0] != g[1] and (g[0] in updated_agents or g[1] in updated_agents)
        ]

        if ON_CLUSTER:
            logger.info(f"About to play {len(to_play)} games on the cluster.")
            run_games_cluster(to_play)
        else:
            logger.info(f"About to play {len(to_play)} games locally.")
            run_games_local(to_play)

        logger.info("Finished game-play round.")


def run_games_cluster(to_play):
    from gridmap import grid_map
    import connectn.results as results
    from connectn.utils import TEMP_DIR

    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir()

    job_temp_dir = TEMP_DIR / time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    job_temp_dir.mkdir()

    logger.info(f"Submitting games to the queue: {to_play}")

    for game_result in grid_map(
        run_game_local,
        to_play,
        mem_free="2G",
        name="conn4match",
        num_slots=1,
        temp_dir=f"{job_temp_dir!s}",
        queue="cognition-all.q",
        add_env={"CREATE_PLOTS": "FALSE", "USE_MEM_FREE": "TRUE"},
        require_cluster=True,
    ):
        results.add_game(game_result)


def run_games_local(to_play):
    import connectn.results as results

    for g in to_play:
        try:
            game_result = run_game_local(*g)
            results.add_game(game_result)
        except Exception:
            logger.exception("This should not happen, unless we are testing")


def run_game_local(
    agent_1: str, agent_2: str, game_seed: Optional[int] = None
) -> GameResult:
    """
    Likely has to be replaced by separate function runnable via the GridEngine
    """
    from connectn.utils import get_size

    rs = np.random.RandomState(game_seed)

    agent_modules = import_agents({})
    agent_names = (agent_1, agent_2)

    def get_name(player: BoardPiece) -> str:
        return agent_names[player - 1]

    states = {agent_name: None for agent_name in agent_names}

    winner = player = NO_PLAYER
    agent_name = agent_1
    results = {PLAYER1: AgentResult(agent_1), PLAYER2: AgentResult(agent_2)}
    gr = GameResult(results[PLAYER1], results[PLAYER2])

    gen_move = {}
    for player, agent_name in zip((PLAYER1, PLAYER2), agent_names):
        try:
            gen_move[agent_name]: GenMove = getattr(
                agent_modules[agent_name], "generate_move"
            )
        except AttributeError:
            results[player].stderr.append(
                "\nYou did not define generate_move at the package level"
            )
            gr.winner = other_player(player)
            results[player].outcome = "FAIL"
            results[gr.winner].outcome = "WIN"
            return gr
        except KeyError as e:
            # If this occurs and it isn't for agent_fail, then something has gone terribly wrong.
            # Presumably one of the agents is not defined in users.py
            logger.exception("Something has gone terribly wrong")
            raise e

    loser_result = "LOSS"
    game_state = initialize_game_state()
    nth_move = 0
    try:
        logger.info(f"Playing game between {agent_1} and {agent_2}")
        moves_q = mp.Manager().Queue()

        end_state = STILL_PLAYING
        playing = True
        action = PlayerAction(0)
        while playing:
            for player, agent_name in zip((PLAYER1, PLAYER2), agent_names):
                move_seed = rs.randint(2 ** 32)
                results[player].seeds.append(move_seed)

                gma = GenMoveArgs(
                    move_seed, game_state.copy(), player, states[agent_name]
                )
                moves_q.put(gma)
                if IS_DEBUGGING:
                    generate_move_process(gen_move[agent_name], moves_q)
                else:
                    ap = mp.Process(
                        target=generate_move_process,
                        args=(gen_move[agent_name], moves_q),
                    )
                    t0 = time.time()
                    ap.start()
                    curr_max_time = 2 * MOVE_TIME_MAX if nth_move < 1 else MOVE_TIME_MAX
                    ap.join(curr_max_time)
                    move_time = time.time() - t0
                    if ap.is_alive():
                        ap.terminate()
                        msg = f"Agent {agent_name} timed out after {curr_max_time} seconds ({move_time:.1f}s)."
                        raise AgentFailed(msg)

                ret: Union[GenMoveSuccess, GenMoveFailure] = moves_q.get(block=True)

                results[player].stdout.append(ret.stdout)
                results[player].stderr.append(ret.stderr)

                if isinstance(ret, GenMoveFailure):
                    error_msg = ret.error_msg
                    msg = f"Agent {agent_name} threw an exception:\n {error_msg}"
                    raise AgentFailed(msg)

                assert isinstance(ret, GenMoveSuccess)
                action = ret.action
                state_size = get_size(ret.state)

                results[player].move_times.append(ret.move_time)
                results[player].state_size.append(state_size)

                if state_size > STATE_MEMORY_MAX:
                    msg = f"Agent {agent_name} used {mib(state_size):.2f} MiB > {mib(STATE_MEMORY_MAX)} MiB"
                    raise AgentFailed(msg)

                if not np.issubdtype(type(action), np.integer):
                    msg = f"Agent {agent_name} returned an invalid type of action {type(action)}"
                    raise AgentFailed(msg)

                action = PlayerAction(action)
                results[player].moves.append(action)
                if not valid_player_action(game_state, action):
                    msg = f"Agent {agent_name} returned an invalid action {action}"
                    raise AgentFailed(msg)

                apply_player_action(game_state, action, player)
                end_state = check_end_state(game_state, player)
                playing = end_state == STILL_PLAYING
                states[agent_name] = ret.state
                if not playing:
                    break
            nth_move += 1
        logger.info(pretty_print_board(game_state))
        for p_i, result in results.items():
            mt = result.move_times
            if len(mt):
                mean_mt = f"mean: {np.mean(mt):.1f}s"
                var_mt = f"var: {np.var(mt):.1f}"
                median_mt = f"median: {np.median(mt):.1f}s"
                max_mt = f"max: {np.max(mt):.1f}s"
                logger.info(
                    f"Move times for {get_name(p_i)} -> {median_mt} {mean_mt} {var_mt} {max_mt}"
                )

        if end_state == IS_WIN:
            winner = player
            logger.info(
                f"Game finished, {get_name(player)} beat {get_name(other_player(player))} by playing column {action}."
            )
        elif end_state == IS_DRAW:
            winner = NO_PLAYER
            logger.info("Game finished, no winner")
        else:
            logger.info("Something went wrong, game-play stopped before the end state.")

    except AgentFailed as err:
        logger.info(pretty_print_board(game_state))
        logger.info(f"Agent failed: {agent_name}")
        logger.info(err)
        winner = other_player(player)
        results[player].stderr.append(str(err))
        loser_result = "FAIL"

    # fig = plt.figure()
    # fig.suptitle('Odds of win')
    # for i, (agent, saved_state) in enumerate(states.items()):
    #     ax = fig.add_subplot(2, 1, i+1)
    #     for odds in saved_state.odds.values():
    #         ax.plot(odds, ('r', 'b')[i])
    #     ax.set_title(agent)
    #     ax.set_ylim(0, 1)
    #
    # fig = plt.figure()
    # fig.suptitle('Odds of draw')
    # for i, (agent, saved_state) in enumerate(states.items()):
    #     ax = fig.add_subplot(2, 1, i+1)
    #     for odds in saved_state.draw.values():
    #         ax.plot(odds, ('r', 'b')[i])
    #     ax.set_title(agent)
    #     ax.set_ylim(0, 1)
    #
    # fig = plt.figure()
    # for i, (agent, saved_state) in enumerate(states.items()):
    #     ax = fig.add_subplot(2, 1, i+1)
    #     ax.plot(saved_state.nodes, label='Nodes')
    #     ax.plot(saved_state.visits, label='Visits')
    #     ax.set_title(agent)
    #     ax.legend()
    #
    # fig = plt.figure()
    # fig.suptitle('Time')
    # for i, (agent, saved_state) in enumerate(states.items()):
    #     ax = fig.add_subplot(2, 1, i+1)
    #     ax.plot(saved_state.time, label=f'{np.mean(saved_state.time):.2f}')
    #     ax.set_title(agent)
    #     ax.legend()
    #
    # plt.show()

    # for i, (agent, saved_state) in enumerate(states.items()):
    #     print(
    #     f'TIME {agent} mu:{np.mean(saved_state.time):.2f},
    #     med:{np.median(saved_state.time):.2f}, max:{np.max(saved_state.time):.2f}'
    #     )

    gr.winner = winner
    if winner == NO_PLAYER:
        results[PLAYER1].outcome = results[PLAYER2].outcome = "DRAW"
    else:
        results[PLAYER1 if winner == PLAYER1 else PLAYER2].outcome = "WIN"
        results[PLAYER2 if winner == PLAYER1 else PLAYER1].outcome = loser_result
    return gr


def generate_move_process(generate_move: GenMove, moves_q: mp.Queue):
    from traceback import StackSummary
    from random import seed as random_seed
    import io
    from time import time
    from contextlib import redirect_stderr, redirect_stdout

    f_stderr, f_stdout = io.StringIO(), io.StringIO()

    gma: GenMoveArgs = moves_q.get()
    np.random.seed(gma.seed)
    random_seed(gma.seed)

    result = None
    try:
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            t0 = time()
            action, saved_state = generate_move(gma.board, gma.player, gma.state)
            move_time = time() - t0
        stdout, stderr = f_stdout.getvalue(), f_stderr.getvalue()
        result = GenMoveSuccess(stdout, stderr, move_time, action, saved_state)
    except Exception as e:
        logger.exception("An exception was thrown by the agent.")
        error_msg = repr(e) + "\n"
        extracted_list = traceback.extract_tb(e.__traceback__)
        for item in StackSummary.from_list(extracted_list).format():
            error_msg += str(item)
        stdout, stderr = f_stdout.getvalue(), f_stderr.getvalue()
        result = GenMoveFailure(stdout, stderr, error_msg)
    finally:
        moves_q.put(result)


def pretty_print_board(board: np.ndarray) -> str:
    bs = "\n|" + "=" * 2 * board.shape[1] + "|\n"
    for i in range(board.shape[0] - 1, -1, -1):
        bs += "|"
        for j in range(board.shape[1]):
            bs += (
                "  "
                if board[i, j] == NO_PLAYER
                else ("X " if board[i, j] == PLAYER1 else "O ")
            )
        bs += "|\n"
    bs += "|" + "=" * 2 * board.shape[1] + "|\n"
    bs += "|"
    for j in range(board.shape[1]):
        bs += str(j) + " "
    bs += "|"

    return bs


def string_to_board(pp_board: str) -> np.ndarray:
    board = initialize_game_state()
    pp_board = pp_board.strip()

    rows = [ln for ln in pp_board.split("|\n|") if "=" not in ln and "0" not in ln]
    assert len(rows) == board.shape[0]
    for row, l in enumerate(rows[::-1]):
        cols = len(l) // 2
        assert cols == board.shape[1]
        for col, p in enumerate(l[::2]):
            if p == "O":
                board[row, col] = PLAYER2
            elif p == "X":
                board[row, col] = PLAYER1
    return board


def initialize_game_state() -> np.ndarray:
    board = np.empty(shape=(CONNECT_N + 2, CONNECT_N + 3), dtype=BoardPiece)
    board.fill(NO_PLAYER)
    return board


def valid_player_action(board: np.ndarray, action: PlayerAction) -> bool:
    return 0 <= action < board.shape[1]


@nb.njit(cache=True)
def other_player(player: BoardPiece) -> BoardPiece:
    return (PLAYER2, PLAYER1)[player - 1]


@nb.njit(cache=True)
def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    if copy:
        board = board.copy()
    for row in np.arange(PlayerAction(board.shape[0])):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player
            return board
    raise AgentFailed("Column was full! ")


@nb.njit(cache=True)
def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    rows, cols = PlayerAction(board.shape[0]), PlayerAction(board.shape[1])

    if last_action is None:
        for row in np.arange(rows):
            for col in np.arange(cols - CONNECT_N + 1):
                if np.all(board[row, col : col + CONNECT_N] == player):
                    return True

        for col in np.arange(cols):
            for row in np.arange(rows - CONNECT_N + 1):
                if np.all(board[row : row + CONNECT_N, col] == player):
                    return True

        diagonal = np.empty(CONNECT_N, dtype=board.dtype)
        for col in np.arange(cols - CONNECT_N + 1):
            for row in np.arange(rows - CONNECT_N + 1):
                for i in np.arange(PlayerAction(CONNECT_N)):
                    diagonal[i] = board[row + i, col + i]

                if np.all(diagonal == player):
                    return True

                for i in np.arange(PlayerAction(CONNECT_N)):
                    diagonal[i] = board[row + CONNECT_N - (i + 1), col + i]

                if np.all(diagonal == player):
                    return True

    else:
        last_col = PlayerAction(last_action)
        last_row = PlayerAction(0)
        for row in np.arange(rows):
            if board[row, last_col] != NO_PLAYER:
                last_row = PlayerAction(row)

        rows_b = max(last_row - CONNECT_N + 1, 0)

        if last_row >= CONNECT_N - 1 and np.all(
            board[rows_b : rows_b + CONNECT_N, last_col] == player
        ):
            return True

        cols_r = min(last_col + CONNECT_N, cols)
        cols_l = max(last_col - CONNECT_N + 1, 0)

        for col in np.arange(cols_l, cols_r - CONNECT_N + 1):
            if np.all(board[last_row, col : col + CONNECT_N] == player):
                return True

        diagonal = np.empty(CONNECT_N, dtype=board.dtype)

        for row, col in zip(
            np.arange(last_row - CONNECT_N + 1, last_row + 1),
            np.arange(last_col - CONNECT_N + 1, last_col + 1),
        ):
            if 0 <= row <= rows - CONNECT_N and 0 <= col <= cols - CONNECT_N:
                b = board[row : row + CONNECT_N, col : col + CONNECT_N]
                for i in np.arange(CONNECT_N):
                    diagonal[i] = b[i, i]

                if np.all(diagonal == player):
                    return True

        for row, col in zip(
            np.arange(last_row + CONNECT_N, last_row - 1, -1),
            np.arange(last_col - CONNECT_N + 1, last_col + 1),
        ):
            if CONNECT_N <= row <= rows and 0 <= col <= cols - CONNECT_N:
                b = board[row - CONNECT_N : row, col : col + CONNECT_N]
                for i in np.arange(CONNECT_N):
                    diagonal[i] = b[CONNECT_N - (i + 1), i]

                if np.all(diagonal == player):
                    return True

    # else:
    #
    #     for row in np.arange(rows):
    #         for col in np.arange(cols - CONNECT_N + 1):
    #             if np.all(board[row, col : col + CONNECT_N] == player):
    #                 return True
    #
    #     for col in np.arange(cols):
    #         for row in np.arange(rows - CONNECT_N + 1):
    #             if np.all(board[row : row + CONNECT_N, col] == player):
    #                 return True
    #
    #     diagonal = np.empty(CONNECT_N, dtype=board.dtype)
    #     for col in np.arange(cols - CONNECT_N + 1):
    #         for row in np.arange(rows - CONNECT_N + 1):
    #             for i in np.arange(PlayerAction(CONNECT_N)):
    #                 diagonal[i] = board[row + i, col + i]
    #
    #             if np.all(diagonal == player):
    #                 return True
    #
    #             for i in np.arange(PlayerAction(CONNECT_N)):
    #                 diagonal[i] = board[row + CONNECT_N - (i + 1), col + i]
    #
    #             if np.all(diagonal == player):
    #                 return True

    return False


@nb.njit(cache=True)
def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> BoardValue:
    """

    Parameters
    ----------
    board : ndarray
    player : BoardPiece
    last_action : PlayerAction, optional

    Returns
    -------
    BoardValue

    """
    if connected_four(board, player, last_action):
        return IS_WIN
    if np.sum(board == NO_PLAYER) == 0:
        return IS_DRAW
    return STILL_PLAYING


def human_vs_agent(generate_move: GenMove):
    def user_move(board, player, saved_state):
        print(pretty_print_board(board))
        col = -1
        while not 0 <= col < board.shape[1]:
            try:
                col = int(input(f'Playing {"X" if player == 1 else "O"}, Col? '))
            except:
                pass
        return col, saved_state

    for play_first in (1, -1):
        saved_state = None
        board = initialize_game_state()
        playing = True
        while playing:
            gen_moves = (generate_move, user_move)[::play_first]
            for player, gen_move in zip((PLAYER1, PLAYER2), gen_moves):
                action, saved_state = gen_move(board.copy(), player, saved_state)
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(f'Player {"X" if player == PLAYER1 else "O"} won')
                    playing = False
                    break
