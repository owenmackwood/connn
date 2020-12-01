import time
import multiprocessing as mp
import numpy as np
import traceback
import logging
from typing import List, Union, Optional, Callable
import connectn.utils as cu
from connectn.users import import_agents, agents
import numba as nb
from scipy.signal.sigtools import _convolve2d


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
        state: Optional[cu.SavedState],
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
        self,
        stdout: str,
        stderr: str,
        move_time: float,
        action: int,
        state: Optional[cu.SavedState],
    ):
        super().__init__(stdout, stderr)
        self.move_time = move_time
        self.action = action
        self.state = state


class GenMoveFailure(GenMoveResult):
    def __init__(self, stdout: str, stderr: str, error_msg: str):
        super().__init__(stdout, stderr)
        self.error_msg = error_msg


def run_games_process(
    rq: mp.Queue, sq: mp.Queue, shutdown: mp.Event, play_all: bool = True
):
    from itertools import product
    from queue import Empty as EmptyQueue
    from threading import Timer, Event
    import connectn.results as cr

    logger = logging.getLogger(__name__)
    logger.info("Started run_games_process.")

    if not cr.GAME_PROCESS_DATA_DIR.exists():
        cr.GAME_PROCESS_DATA_DIR.mkdir()

    repetitions = 1
    run_all_after = (
        60.0 * 60 * cu.RUN_ALL_EVERY
    )  # Run all-against-all every cu.RUN_ALL_EVERY hours
    block_time = 1

    agent_modules = import_agents({})
    cr.initialize(agent_modules.keys())

    if play_all:
        logger.info(f"Just started, running all-against-all once.")
        updated_agents = list(agents())
    else:
        logger.info("Skipping all-against-all.")
        updated_agents = []

    time_to_play_all = Event()
    timer = Timer(run_all_after, lambda ev: ev.set(), args=(time_to_play_all,))
    timer.start()
    while not shutdown.is_set():
        if updated_agents:
            agent_modules = import_agents(agent_modules)
            agent_modules.pop("agent_fail", None)

            to_play = repetitions * [
                list(g)  # grid_map wants args as a list
                for g in product(agent_modules.keys(), agent_modules.keys())
                if g[0] != g[1] and (g[0] in updated_agents or g[1] in updated_agents)
            ]
            updated_agents.clear()
            if cu.ON_CLUSTER:
                logger.info(f"About to play {len(to_play)} games on the cluster.")
                run_games_cluster(to_play)
            else:
                logger.info(f"About to play {len(to_play)} games locally.")
                run_games_local(to_play)

            logger.info("Finished game-play round.")

            played = set(n for p in to_play for n in p if cr.record_games_for_agent(n))

            new_results = {}
            for agent_name in played:
                with open(f"{cr.agent_games_file_path(agent_name)}", "rb") as f:
                    new_results[agent_name] = f.read()
            with open(f"{cr.RESULTS_FILE_PATH!s}", "rb") as f:
                new_results[cu.TOURNAMENT_FILE] = f.read()
            logger.info(
                f"Sending {len(new_results)} modified result files to the server."
            )
            sq.put(new_results)

        try:
            # Check the message queue for updated agents
            q_data = rq.get(block=True, timeout=block_time)
        except EmptyQueue:
            if time_to_play_all.is_set():
                time_to_play_all.clear()
                timer = Timer(
                    run_all_after, lambda ev: ev.set(), args=(time_to_play_all,)
                )
                timer.start()
                updated_agents = list(agents())
                logger.info("Timed to run all-against-all.")
        else:
            if isinstance(q_data, str) and q_data == "PLAY_ALL":
                updated_agents = list(agents())
                msg = "Received request to play all-against-all."
            else:
                updated_agents = cu.update_user_agent_code(q_data)
                msg = f'Received {len(updated_agents)} updated agents for game-play: {" ".join(updated_agents)}'
            logger.info(msg)

    timer.cancel()
    logger.info(f"Shutting down run_games gracefully.")


def run_games_cluster(to_play: List[List[str]]):
    from gridmap import grid_map
    import connectn.results as results
    from connectn.utils import TEMP_DIR

    logger = logging.getLogger(__name__)

    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir()

    job_temp_dir = TEMP_DIR / time.strftime("%Y-%m-%d-%Hh%Mm%Ss")
    job_temp_dir.mkdir()

    logger.info(f"Submitting games to the queue: {to_play}")

    n_games = len(to_play)
    n_done = 1
    for game_result in grid_map(
        run_single_game,
        to_play,
        mem_free="2G",
        name="conn4match",
        num_slots=1,
        temp_dir=f"{job_temp_dir!s}",
        queue="cognition-all.q",
        add_env={"CREATE_PLOTS": "FALSE", "USE_MEM_FREE": "TRUE"},
        require_cluster=True,
    ):
        logging.info(f"Received result {n_done} of {n_games}")
        results.add_game(game_result)
        logging.info(f"Wrote result {n_done} of {n_games} to disk.")
        n_done += 1

    logging.info(f"Finished all {n_games} games.")


def run_games_local(to_play: List[List[str]]):
    import connectn.results as results

    logger = logging.getLogger(__name__)

    for g in to_play:
        try:
            game_result = run_single_game(*g)
            results.add_game(game_result)
        except Exception:
            logger.exception("This should not happen, unless we are testing")


def run_single_game(
    agent_1: str, agent_2: str, game_seed: Optional[int] = None
) -> GameResult:
    """
    Likely has to be replaced by separate function runnable via the GridEngine
    """
    from connectn import IS_DEBUGGING
    from queue import Empty as EmptyQueue

    logger = logging.getLogger(__name__)
    logger.debug(f"Entered run_single_game for {agent_1} vs {agent_2}")
    rs = np.random.RandomState(game_seed)

    agent_modules = import_agents({})
    agent_names = (agent_1, agent_2)

    def get_name(_player: BoardPiece) -> str:
        return agent_names[_player - 1]

    states = {agent_name: None for agent_name in agent_names}

    winner = player = NO_PLAYER
    agent_name = agent_1
    results = {PLAYER1: AgentResult(agent_1), PLAYER2: AgentResult(agent_2)}
    gr = GameResult(results[PLAYER1], results[PLAYER2])

    gen_move = {}
    for player, agent_name in zip((PLAYER1, PLAYER2), agent_names):
        try:
            gen_move[agent_name]: cu.GenMove = getattr(
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

    game_state = initialize_game_state()
    for player, agent_name in zip((PLAYER1, PLAYER2), agent_names):
        try:
            init = getattr(agent_modules[agent_name], "initialize")
            init(game_state.copy(), player)
        except (Exception, AttributeError):
            pass

    loser_result = "LOSS"
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
                    ap.join(cu.MOVE_TIME_MAX)
                    move_time = time.time() - t0
                    if ap.is_alive():
                        ap.terminate()
                        loser_result = "TIMEOUT"
                        msg = f"Agent {agent_name} timed out after {cu.MOVE_TIME_MAX} seconds ({move_time:.1f}s)."
                        raise AgentFailed(msg)

                try:
                    ret: Union[GenMoveSuccess, GenMoveFailure] = moves_q.get(
                        block=True, timeout=60.0
                    )
                except EmptyQueue:
                    logger.exception("Timed out waiting to get move result from queue")
                    raise

                results[player].stdout.append(ret.stdout)
                results[player].stderr.append(ret.stderr)

                if isinstance(ret, GenMoveFailure):
                    loser_result = "EXCEPTION"
                    error_msg = ret.error_msg
                    msg = f"Agent {agent_name} threw an exception:\n {error_msg}"
                    raise AgentFailed(msg)

                assert isinstance(ret, GenMoveSuccess)
                action = ret.action
                state_size = cu.get_size(ret.state)

                results[player].move_times.append(ret.move_time)
                results[player].state_size.append(state_size)

                if state_size > cu.STATE_MEMORY_MAX:
                    loser_result = "MAX_STATE_MEM"
                    msg = f"Agent {agent_name} used {cu.mib(state_size):.2f} MiB > {cu.mib(cu.STATE_MEMORY_MAX)} MiB"
                    raise AgentFailed(msg)

                if not np.issubdtype(type(action), np.integer):
                    loser_result = "NONINT_ACTION"
                    msg = f"Agent {agent_name} returned an invalid type of action {type(action)}"
                    raise AgentFailed(msg)

                action = PlayerAction(action)
                results[player].moves.append(action)
                if not valid_player_action(game_state, action):
                    loser_result = "INVALID_ACTION"
                    msg = f"Agent {agent_name} returned an invalid action {action}"
                    raise AgentFailed(msg)

                apply_player_action(game_state, action, player)
                end_state = check_end_state(game_state, player)
                playing = end_state == STILL_PLAYING
                states[agent_name] = ret.state
                if not playing:
                    break

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

    logger.debug(f"Finished run_single_game for {agent_1} vs {agent_2}")

    return gr


def generate_move_process(generate_move: cu.GenMove, moves_q: mp.Queue):
    from traceback import StackSummary
    from random import seed as random_seed
    import io
    from time import time
    import pickle
    from contextlib import redirect_stderr, redirect_stdout

    logger = logging.getLogger(__name__)
    f_stderr, f_stdout = io.StringIO(), io.StringIO()

    gma: GenMoveArgs = moves_q.get()
    np.random.seed(gma.seed)
    random_seed(gma.seed)

    try:
        with redirect_stdout(f_stdout), redirect_stderr(f_stderr):
            t0 = time()
            returned = generate_move(gma.board, gma.player, gma.state)
            saved_state = None
            if isinstance(returned, tuple):
                action = returned[0]
                if len(returned) > 1:
                    saved_state = returned[1]
            else:
                action = returned

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

    try:
        moves_q.put(result)
    except pickle.PickleError:
        logger.exception(
            "Internal error in trying to send the result, probably caused by saved_state"
        )
        moves_q.put(GenMoveSuccess(stdout, stderr, move_time, action, None))


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
def get_last_row_played(board, last_action):
    last_row = 0
    while last_row < board.shape[0] and board[last_row, last_action] != NO_PLAYER:
        last_row += 1
    return last_row - 1


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
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, reduction = False,
) -> bool:
    """
    Time to check ~20k boards. Random games are played
    until win or draw, every intermediate board is included in set.
    Board stats -> mean pieces: 12.3, median: 11.0

    Last action: True,
        reduction: True: 6.6e-02, 21358 boards (3.1 us / board)
        reduction: False: 3.3e-02, 21440 boards (1.5 us / board)

    Last action: False,
        reduction: True: 1.2e-01, 21358 boards (5.6 us / board)
        reduction: False: 7.8e-02, 21358 boards (3.7 us / board)

    :param board:
    :param player:
    :param last_action:
    :param reduction:
    :return:
    """
    rows, cols = PlayerAction(board.shape[0]), PlayerAction(board.shape[1])
    bb = board == player
    if reduction:
        reduced_value = 2 ** (CONNECT_N - 1)
        bb = bb.astype(np.int8)

        def dr(vec):
            for _ in range(CONNECT_N - 1):
                vec = vec[1:] + vec[:-1]
            return np.any(vec == reduced_value)

        if last_action is None:
            c, r = bb.copy(), bb.copy()
            for _ in range(CONNECT_N - 1):
                c = c[1:, :] + c[:-1, :]
                r = r[:, 1:] + r[:, :-1]
            if np.any(c == reduced_value) or np.any(r == reduced_value):
                return True
            # Using the arrays as above is faster
            # for i in range(rows):
            #     if dr(bb[i, :]):
            #         return True
            # for j in range(cols):
            #     if dr(bb[:, j]):
            #         return True

            for dia_k in range(-(rows - CONNECT_N), (cols - CONNECT_N) + 1):
                if dr(np.diag(bb, dia_k)) or dr(np.diag(bb[::-1, :], dia_k)):
                    return True
        else:
            last_row = get_last_row_played(board, last_action)

            ret = (dr(bb[last_row, :]) or
                   dr(bb[:, last_action]) or
                   dr(np.diag(bb, k=last_action - last_row)) or
                   dr(np.diag(bb[::-1, :], k=last_action - (rows - last_row - 1)))
                   )

            return ret

    else:
        if last_action is None:
            rows_edge = rows - CONNECT_N + 1
            cols_edge = cols - CONNECT_N + 1

            # Adding early breaks did not speed things up (on average over many random boards)
            # might have actually slowed it a tiny bit.
            for i in range(rows):
                for j in range(cols_edge):
                    if np.all(bb[i, j:j + CONNECT_N]):
                        return True

            for j in range(cols):
                for i in range(rows_edge):
                    if np.all(bb[i:i + CONNECT_N, j]):
                        return True

            for i in range(rows_edge):
                for j in range(cols_edge):
                    block = bb[i:i + CONNECT_N, j:j + CONNECT_N]
                    if np.all(np.diag(block)) or np.all(np.diag(block[::-1, :])):
                        return True

        else:
            last_row = get_last_row_played(board, last_action)

            for j in range(cols - CONNECT_N + 1):
                if np.all(bb[last_row, j : j + CONNECT_N]):
                    return True

            for j in range(rows - CONNECT_N + 1):
                if np.all(bb[j : j + CONNECT_N, last_action]):
                    return True

            dia = np.diag(bb, k=last_action - last_row)
            for i in range(dia.size - CONNECT_N + 1):
                if np.all(dia[i : i + CONNECT_N]):
                    return True

            dia = np.diag(bb[::-1, :], k=last_action  - (rows - last_row - 1))
            for i in range(dia.size - CONNECT_N + 1):
                if np.all(dia[i: i + CONNECT_N]):
                    return True

    return False


def connected_four_generators(
    board: np.ndarray, player: BoardPiece, last_action: PlayerAction
) -> bool:
    """
    Takes 115 us / board, which is about the same as the `connected_four` above if it isn't compiled (1.5 us otherwise).
    """
    rows, cols = PlayerAction(board.shape[0]), PlayerAction(board.shape[1])
    last_row = get_last_row_played(board, last_action)
    bb = board == player

    return (any(np.all(bb[last_row, j: j + CONNECT_N]) for j in range(cols - CONNECT_N + 1)) or
            any(np.all(bb[j: j + CONNECT_N, last_action]) for j in range(rows - CONNECT_N + 1)) or
            any(np.all(dia[i: i + CONNECT_N])
                    for dia in (np.diag(bb, k=last_action - last_row),
                                np.diag(bb[::-1, :], k=last_action - (rows - last_row - 1)))
                        for i in range(dia.size - CONNECT_N + 1)))


col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])


def connected_four_convolve(board: np.ndarray, player: BoardPiece,) -> bool:
    # No point in using last_action here because it isn't any faster.
    # rows, cols = board.shape
    #
    # if last_action is not None:
    #     lc = last_action
    #     last_rows = np.atleast_1d(np.argwhere(board[:, lc] != NO_PLAYER).squeeze())
    #     lr = last_rows[-1] if len(last_rows) else 0
    #     min_row = max(lr-CONNECT_N+1, 0)
    #     max_row = min(lr+CONNECT_N, rows)
    #     min_col = max(lc-CONNECT_N+1, 0)
    #     max_col = min(lc+CONNECT_N, cols)
    #     board = board[min_row : max_row, min_col : max_col]
    #     # board = np.array(board)
    #     # print(last_rows, view_rows, view_cols, board.shape)
    #     # print(pretty_print_board(board))
    # else:
    board = board.copy()

    board[board == other_player(player)] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True
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


def user_move(board, _player, saved_state):
    col = -1
    while not 0 <= col < board.shape[1]:
        try:
            col = int(input(f"Col? "))
        except:
            pass
    return col, saved_state


def human_vs_agent(
    generate_move_1: cu.GenMove,
    generate_move_2: cu.GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {"X" if player == 1 else "O"}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        )
                    playing = False
                    break


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    benchmark = True
    if benchmark:
        import timeit
        compare_reduct = True
        if compare_reduct:
            number = 1000
            games = 1000
            all_boards = []
            sizes = []
            empty_board = initialize_game_state()
            for game_n in range(games):
                board = empty_board.copy()
                curr_player = PLAYER1
                end_state = check_end_state(board, curr_player)
                move_n = 0
                while end_state == STILL_PLAYING:
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
                    sizes.append((board != NO_PLAYER).sum())
                    all_boards.append((board, curr_player, move,))
            sizes = np.array(sizes)
            print(f"Boards: {np.mean(sizes)} {np.median(sizes)} {(sizes == np.product(empty_board.shape)).sum()}")
            for la, reduction in ((True, False), (False, False)):  #(True, True), (False, True),
                ns = {
                    "cf": connected_four,
                    "all_boards": all_boards,
                    "board": empty_board,
                    "PLAYER1": PLAYER1,
                    "r": reduction,
                    "la": la,
                }
                ti0 = timeit.timeit(
                    "[cf(b, p, last_action=m if la else None, reduction=r) for b, p, m in all_boards]",
                    "cf(board, PLAYER1)",
                    globals=ns,
                    number=number,
                )
                tot_time = ti0 / number
                per_board = tot_time / len(all_boards)
                print(f"Last action: {la}, reduction: {reduction}:  {tot_time:.1e} sec total, {len(all_boards)} boards ({per_board*1e6:.3} us per board)")
            # for fn in (connected_four, connected_four_ultimate):
            #     ns = {
            #         "cf": fn,
            #         "all_boards": all_boards,
            #         "board": empty_board,
            #         "PLAYER1": PLAYER1,
            #     }
            #     ti0 = timeit.timeit(
            #         "[cf(b, p, m) for b, p, m in all_boards]",
            #         "cf(board, PLAYER1, 0)",
            #         globals=ns,
            #         number=number,
            #     )
            #     tot_time = ti0 / number
            #     per_board = tot_time / len(all_boards)
            #     print(f"{fn}: {tot_time:.1e} sec total, {len(all_boards)} boards ({per_board*1e6:.3} us per board)")

            plt.hist(sizes, bins=41)
            plt.show()

        else:
            _action = PlayerAction(3)
            number = 10000
            _board = initialize_game_state()
            for _col in np.arange(_board.shape[1], step=2):
                _board[: CONNECT_N - 1, _col] = PLAYER1
                _board[CONNECT_N - 1 :, _col] = PLAYER2
            for _col in np.arange(1, _board.shape[1] - 1, step=2):
                _board[CONNECT_N - 1 :, _col] = PLAYER1
                _board[: CONNECT_N - 1, _col] = PLAYER2

            for _col in range(7):
                ns = {
                    "connected_four_convolve": connected_four_convolve,
                    "board": _board,
                    "PLAYER1": PLAYER1,
                }
                ti0 = timeit.timeit(
                    "connected_four_convolve(board, PLAYER1)",
                    "connected_four_convolve(board, PLAYER1)",
                    globals=ns,
                    number=number,
                )
                print(f"Conv version: {ti0 / number:.1e}")

                ns = {"connected_four": connected_four, "board": _board, "PLAYER1": PLAYER1}
                ti0 = timeit.timeit(
                    "connected_four(board, PLAYER1)",
                    "connected_four(board, PLAYER1)",
                    globals=ns,
                    number=number,
                )
                print(f"Dumb version: {ti0 / number:.1e}")

                ns = {
                    "connected_four": connected_four,
                    "board": _board,
                    "PLAYER1": PLAYER1,
                    "action": _action,
                }
                ti0 = timeit.timeit(
                    "connected_four(board, PLAYER1, action)",
                    "connected_four(board, PLAYER1, action)",
                    globals=ns,
                    number=number,
                )
                print(f"Smrt version: {ti0 / number:.1e}")
                print("=" * 10)
    else:
        human_vs_agent(user_move)
