import os
import time
import multiprocessing as mp
import numpy as np
from typing import List
from connectn.utils import GenMove, SavedState, nb
from connectn.users import import_agents, agents
from connectn.utils import MOVE_TIME_MAX, STATE_MEMORY_MAX, ON_CLUSTER
from connectn.utils import ARCHIVE_FORMAT, IS_DEBUGGING

BOARD_SHAPE = (6, 7)
btype = np.int8
CONNECT_N = 4
IS_WIN = btype(1)
STILL_PLAYING = btype(0)
IS_DRAW = btype(-1)
EMPTY = btype(0)
PLAYER1 = btype(1)
PLAYER2 = btype(2)


class AgentFailed(Exception):
    pass


def run_games(q: mp.Queue):
    from itertools import product
    import connectn.agents as cna
    import shutil
    import connectn.results as results

    agent_modules = import_agents({})
    results.initialize(agent_modules.keys())
    # agent_modules = dict()
    startup = True
    while True:
        # Check the message queue for updated agents
        if startup:
            startup = True
            updated_agents = list(agents())
            print(f'Just started, running all-against-all once.')
        else:
            print('Game-playing process entering queue to wait for new agents')
            q_data = q.get(block=True)  #, timeout=10*60)
            if isinstance(q_data, str) and q_data == 'SHUTDOWN':
                return
            updated_agent_archives = q_data
            for agent_name, archive_path in updated_agent_archives:
                module_path = os.path.join(cna.__path__[0], agent_name)
                if os.path.exists(module_path):
                    shutil.rmtree(module_path)
                os.makedirs(module_path)
                shutil.unpack_archive(archive_path, module_path, ARCHIVE_FORMAT)
                os.remove(archive_path)
                results.add_agent(agent_name)

            updated_agents = [agent_name for agent_name, _ in updated_agent_archives]
            print(f'Received {len(updated_agents)} updated agents for game-play: {" ".join(updated_agents)}')

        agent_modules = import_agents(agent_modules)

        to_play = [g for g in product(agent_modules.keys(), agent_modules.keys())
                   if g[0] != g[1] and (g[0] in updated_agents or g[1] in updated_agents)]

        print(f'About to play {len(to_play)} games.')
        for g in to_play:
            game_result = run_game(*g)
            results.add_game(game_result)
        print('Finished game-play round.')


def run_game_cluster(agent_1: str, agent_2: str):
    # import connectn
    # connectn.__path__
    print(f'Submitting game between {agent_1} and {agent_2} to the queue.')


def run_game_local(agent_1: str, agent_2: str):
    '''
    Likely has to be replaced by separate function runnable via the GridEngine
    '''
    import matplotlib.pyplot as plt
    agent_modules = import_agents({})
    agent_names = (agent_1, agent_2)
    states = {agent_name:None for agent_name in agent_names}

    winner = player = btype(0)
    agent_name = agent_1
    results = {agent_name:AgentResult(agent_name, []) for agent_name in agent_names}
    gr = GameResult(results[agent_1], results[agent_2])

    gen_move = {}
    for player, agent_name in zip((PLAYER1, PLAYER2), agent_names):
        try:
            gen_move[agent_name] = agent_modules[agent_name].generate_move
        except AttributeError:
            results[agent_name].stderr += 'You did not define generate_move at the package level'
            gr.winner = other_player(player)
            results[agent_name].outcome = 'FAIL'
            results[agent_names[gr.winner-1]].outcome = 'WIN'
            return gr

    board = np.zeros(shape=BOARD_SHAPE, dtype=btype)
    try:
        print(f'Playing game between {agent_1} and {agent_2}')
        moves_q = mp.Manager().Queue()

        end_state = 0
        playing = True
        while playing:
            for a_i, agent_name in enumerate(agent_names, 1):
                player = btype(a_i)
                moves_q.put((board.copy(), player, states[agent_name]))
                if IS_DEBUGGING:
                    generate_move_process(gen_move[agent_name], moves_q)
                    status, final_memory, action, state_n = moves_q.get()
                else:
                    ap = mp.Process(target=generate_move_process,
                                    args=(gen_move[agent_name], moves_q))
                    ap.start()
                    ap.join(MOVE_TIME_MAX)
                    status, final_memory, action, state_n = moves_q.get()#block=True, timeout=MOVE_TIME_MAX)
                    if ap.is_alive():
                        ap.terminate()
                        raise AgentFailed(f"Agent {agent_name} timed out after {MOVE_TIME_MAX} seconds.")


                if status != 0:
                    raise AgentFailed(f"Agent {agent_name} threw an exception")
                elif final_memory > STATE_MEMORY_MAX:
                    raise AgentFailed(f"Agent {agent_name} used {final_memory//2**20} MB > {STATE_MEMORY_MAX//2**20} MB")
                elif not isinstance(action, (int, np.int8, np.int16, np.int32, np.int64)):
                    raise AgentFailed(f"Agent {agent_name} returned an invalid type of action {type(action)}")

                results[agent_name].moves.append(action)

                if not (0 <= action < BOARD_SHAPE[1]):
                    raise AgentFailed(f"Agent {agent_name} returned an invalid column {action}")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                playing = end_state == 0
                states[agent_name] = state_n
                if not playing:
                    break
        print(pretty_print_board(board))
        if end_state == IS_WIN:
            winner = player
            print(f"Game finished, {agent_names[player-1]} won by playing column {action}.")
        elif end_state == IS_DRAW:
            winner = 0
            print("Game finished, no winner")
        else:
            print("Something went wrong, game-play stopped before the end state.")

    except AgentFailed as err:
        print(pretty_print_board(board))
        print(f'Agent failed: {agent_name}')
        print(err)
        winner = other_player(player)
        results[agent_name].stderr += str(err)
        results[agent_name].outcome = 'FAIL'
        results[agent_names[winner-1]].outcome = 'WIN'

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
    #     print(f'TIME {agent} mu:{np.mean(saved_state.time):.2f}, med:{np.median(saved_state.time):.2f}, max:{np.max(saved_state.time):.2f}')

    gr.winner = winner
    if winner == 0:
        results[agent_1].outcome = 'DRAW'
        results[agent_2].outcome = 'DRAW'
    elif winner == 1:
        results[agent_1].outcome = 'WIN'
        results[agent_2].outcome = 'LOSS'
    elif winner == 2:
        results[agent_1].outcome = 'LOSS'
        results[agent_2].outcome = 'WIN'
    return gr


def generate_move_process(generate_move: GenMove, moves_q: mp.Queue):
    # import psutil
    # process = psutil.Process(os.getpid())
    from connectn.utils import get_size
    import traceback
    board, player, saved_state = moves_q.get()
    action = -1
    size = -1
    np.random.seed(None)
    try:
        action, saved_state = generate_move(board, player, saved_state)
        # print('Agented returned, estimating state size')
        size = get_size(saved_state)
        # print(f'Saved state was {size/2**20:.3f} MB')
        status = 0
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        status = 1

    moves_q.put((status, size, action, saved_state))


def pretty_print_board(board: np.ndarray):
    bs = '|' + '='*2*board.shape[1] + '|\n'
    for i in range(board.shape[0]-1, -1, -1):
        bs+= '|'
        for j in range(board.shape[1]):
            bs += '  ' if board[i,j] == 0 else ('X ' if board[i,j] == 1 else 'O ')
        bs += '|\n'
    bs += '|' + '='*2*board.shape[1] + '|\n'
    bs += '|'
    for j in range(board.shape[1]):
        bs += str(j) + ' '
    bs += '|'

    return bs


@nb.njit(cache=True)
def other_player(player: btype) -> btype:
    return (PLAYER2, PLAYER1)[player - 1]


@nb.njit(cache=True)
def apply_player_action(board: np.ndarray, col: btype, player: btype, copy: bool = False):
    if copy:
        board = board.copy()
    for row in np.arange(btype(board.shape[0])):
        if board[row, col] == EMPTY:
            board[row, col] = player
            return board
    raise AgentFailed('Column was full! ')


@nb.njit(cache=True)
def connected_four(board: np.ndarray, player: btype) -> bool:
    rows, cols = btype(board.shape[0]), btype(board.shape[1])

    for row in np.arange(rows):
        for col in np.arange(cols-3):
            if np.all(board[row, col:col+4] == player):
                return True

    for col in np.arange(cols):
        for row in np.arange(rows-3):
            if np.all(board[row:row+4, col] == player):
                return True

    for col in np.arange(cols-3):
        for row in np.arange(rows-3):
            found = True
            for i in np.arange(btype(4)):
                if board[row+i, col+i] != player:
                    found = False
                    break
            if found:
                return True

            found = True
            for i in np.arange(btype(4)):
                if board[row+(3-i), col+i] != player:
                    found = False
                    break
            if found:
                return True

    return False


@nb.njit(cache=True)
def check_end_state(board: np.ndarray, player: btype) -> btype:
    '''
    :param board:
    :return: 1 if winning state, 0 if still playing, -1 if board is full
    '''
    if connected_four(board, player):
        return IS_WIN
    if np.sum(board == EMPTY) == 0:
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
        board = np.zeros(BOARD_SHAPE, dtype=btype)
        playing = True
        while playing:
            gen_moves = (generate_move, user_move)[::play_first]
            for player, gen_move in zip((PLAYER1, PLAYER2), gen_moves):
                player = btype(player)
                action, saved_state = gen_move(board, player, saved_state)
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == IS_DRAW:
                        print('Game ended in draw')
                    else:
                        print(f'Player {"X" if player == PLAYER1 else "O"} won')
                    playing = False
                    break


class AgentResult:
    def __init__(self, agent_name: str, moves: List[int], stdout: str='', stderr: str=''):
        self.name: str = agent_name
        self.moves: List[int] = moves
        self.stdout: str = stdout
        self.stderr: str = stderr
        self.outcome: str = 'NONE'


class GameResult:
    def __init__(self, agent_r1: AgentResult, agent_r2: AgentResult):
        self.result_1: AgentResult = agent_r1
        self.result_2: AgentResult = agent_r2
        self.winner: int = -1
        self.time_sec: float = time.time()
        self.time_str: str = time.ctime()


if ON_CLUSTER:
    run_game = run_game_cluster
else:
    run_game = run_game_local
