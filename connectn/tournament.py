import traceback
import logging
import time
import numpy as np
import multiprocessing as mp
from typing import List, Union, Optional
import connectn.utils as cu
from connectn.users import import_agents, agents
from connectn.game import PlayerAction, BoardPiece, AgentFailed
from connectn.game import NO_PLAYER


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


def run_tournament_process(
    rq: mp.Queue, sq: mp.Queue, shutdown: mp.Event, play_all: bool = True
):
    from itertools import product
    from queue import Empty as EmptyQueue
    from threading import Timer, Event
    import connectn.results as cr

    logger = logging.getLogger(__name__)
    logger.info("Started run_tournament_process.")

    if not cr.TOURNAMENT_PROCESS_DATA_DIR.exists():
        cr.TOURNAMENT_PROCESS_DATA_DIR.mkdir()

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
                run_tournament_cluster(to_play)
            else:
                logger.info(f"About to play {len(to_play)} games locally.")
                run_tournament_local(to_play)

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
    logger.info(f"Shutting down run_tournament_process gracefully.")


def run_tournament_cluster(to_play: List[List[str]]):
    from gridmap import grid_map
    import connectn.results as results
    from connectn.utils import TEMP_DIR

    logger = logging.getLogger(__name__)

    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True)

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


def run_tournament_local(to_play: List[List[str]]):
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
    from connectn.game import initialize_game_state, other_player
    from connectn.game import valid_player_action, apply_player_action
    from connectn.game import check_end_state, pretty_print_board
    from connectn.game import PLAYER1, PLAYER2, GameStatus

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

        end_state = GameStatus.STILL_PLAYING
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
                playing = end_state == GameStatus.STILL_PLAYING
                states[agent_name] = ret.state
                if not playing:
                    break

        if end_state == GameStatus.IS_WIN:
            winner = player
            logger.info(
                f"Game finished, {get_name(player)} beat {get_name(other_player(player))} by playing column {action}."
            )
        elif end_state == GameStatus.IS_DRAW:
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
