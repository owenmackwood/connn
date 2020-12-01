import time
import numpy as np
from typing import Optional, Callable
from enum import Enum
import connectn.utils as cu
import numba as nb
from scipy.signal.sigtools import _convolve2d

CONNECT_N = np.int8(4)
PlayerAction = np.int8
class GameStatus(Enum):
    IS_WIN = 1
    STILL_PLAYING = 0
    IS_DRAW = -1

BoardPiece = np.int8
NO_PLAYER = BoardPiece(0)
PLAYER1 = BoardPiece(1)
PLAYER2 = BoardPiece(2)


class AgentFailed(Exception):
    pass


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
    last_row = board.shape[0] - 1
    while last_row > 0 and board[last_row, last_action] == NO_PLAYER:
        last_row -= 1
    return last_row


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
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, reduction: bool = False,
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
                if np.all(bb[last_row, j:j + CONNECT_N]):
                    return True

            for j in range(rows - CONNECT_N + 1):
                if np.all(bb[j:j + CONNECT_N, last_action]):
                    return True

            dia = np.diag(bb, k=last_action - last_row)
            for i in range(dia.size - CONNECT_N + 1):
                if np.all(dia[i:i + CONNECT_N]):
                    return True

            dia = np.diag(bb[::-1, :], k=last_action - (rows - last_row - 1))
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
) -> GameStatus:
    """

    Parameters
    ----------
    board : ndarray
    player : BoardPiece
    last_action : PlayerAction, optional

    Returns
    -------
    GameStatus

    """
    if connected_four(board, player, last_action):
        return GameStatus.IS_WIN
    if np.sum(board == NO_PLAYER) == 0:
        return GameStatus.IS_DRAW
    return GameStatus.STILL_PLAYING


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
                if end_state != GameStatus.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameStatus.IS_DRAW:
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
                _board = empty_board.copy()
                curr_player = PLAYER1
                es = check_end_state(_board, curr_player)
                move_n = 0
                while es == GameStatus.STILL_PLAYING:
                    move_n += 1
                    curr_player = other_player(curr_player)
                    moves = np.array(
                        [
                            col
                            for col in np.arange(PlayerAction(_board.shape[1]))
                            if _board[PlayerAction(-1), col] == NO_PLAYER
                        ]
                    )
                    move = np.random.choice(moves, 1)[0]
                    apply_player_action(_board, move, curr_player)
                    es = check_end_state(_board, curr_player)
                    sizes.append((_board != NO_PLAYER).sum())
                    all_boards.append((_board, curr_player, move,))
            sizes = np.array(sizes)
            print(f"Boards: {np.mean(sizes)} {np.median(sizes)} {(sizes == np.product(empty_board.shape)).sum()}")
            for la, re in ((True, False), (False, False)):  # (True, True), (False, True),
                ns = {
                    "cf": connected_four,
                    "all_boards": all_boards,
                    "board": empty_board,
                    "PLAYER1": PLAYER1,
                    "r": re,
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

                print(f"Last action: {la}, reduction: {re}:  {tot_time:.1e} sec total, "
                      f"{len(all_boards)} boards ({per_board*1e6:.3} us per board)")
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
                _board[CONNECT_N - 1:, _col] = PLAYER2
            for _col in np.arange(1, _board.shape[1] - 1, step=2):
                _board[CONNECT_N - 1:, _col] = PLAYER1
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
