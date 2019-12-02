import os
import time
import tables
import numpy as np
from tables import IsDescription, Int32Col, StringCol, Float64Col, BoolCol
from connectn.utils import DATA_DIR
from connectn.game import GameResult
from typing import Iterable

RESULTS_FILE_PATH = DATA_DIR / "results.h5"
name_size = 32

"""
agents
    current -> Name, Current version, Current ELO rank, Last upload date / time
    group_a -> Version, Date uploaded, Games won, Games lost, Games failed, ELO rank
    ...
    agent_random
    
games

"""


class TimeStamp(IsDescription):
    time_str = StringCol(25, pos=0)
    time_sec = Float64Col(pos=1)


class CurrentAgentRow(IsDescription):
    name = StringCol(name_size, pos=0)
    version = Int32Col(pos=1)
    rating = Float64Col(pos=3, dflt=0.0)
    uploaded = TimeStamp()


class AgentVersionRow(IsDescription):
    version = Int32Col(pos=0, dflt=0)
    won = Int32Col(pos=1, dflt=0)
    lost = Int32Col(pos=2, dflt=0)
    drawn = Int32Col(pos=3, dflt=0)
    failed = Int32Col(pos=4, dflt=0)
    rating = Float64Col(pos=5, dflt=0.0)
    uploaded = TimeStamp()


"""
agent
    <version number>
        games -> Date, Time, Opponent, Opponent version, Played first (bool), Outcome, Game key
        <game key>
            moves -> Player 1 move, Player 2 move
            stdout -> string array
            stderr -> string array
"""


class GameOutcomeRow(IsDescription):
    key = Int32Col(pos=0)
    opponent = StringCol(name_size, pos=1)
    version = Int32Col(pos=2)
    moved_first = BoolCol(pos=3)
    outcome = StringCol(5, pos=4)
    when = TimeStamp()


def add_agent(agent_name: str):
    with tables.open_file(str(RESULTS_FILE_PATH), "a") as f:
        try:
            av_table = f.get_node("/agents", agent_name)
        except tables.NoSuchNodeError:
            av_table = f.create_table(
                "/agents", agent_name, AgentVersionRow, createparents=True
            )

        n_versions = av_table.nrows
        t_uploaded = time.time()
        t_str = time.ctime()
        av_row = av_table.row
        av_row["version"] = n_versions
        av_row["uploaded/time_str"] = t_str
        av_row["uploaded/time_sec"] = t_uploaded
        av_row.append()
        av_table.flush()

        found = False
        for ac_row in f.root.current.where(f'(name == b"{agent_name}")'):
            assert (
                not found
            ), "Somehow there was more than one row with the same agent name"
            found = True
            ac_row["version"] = n_versions
            ac_row["uploaded/time_str"] = t_str
            ac_row["uploaded/time_sec"] = t_uploaded
            ac_row.update()

        if not found:
            ac_row = f.root.current.row
            ac_row["name"] = agent_name
            ac_row["version"] = n_versions
            ac_row["uploaded/time_str"] = t_str
            ac_row["uploaded/time_sec"] = t_uploaded
            ac_row.append()

        f.flush()


def get_agent_version(agent_name: str):
    with tables.open_file(str(RESULTS_FILE_PATH), "r") as f:
        found = False
        version = -1
        for ac_row in f.root.current.where(f'(name == b"{agent_name}")'):
            assert (
                not found
            ), "Somehow there was more than one row with the same agent name"
            found = True
            version = ac_row["version"]
    return version


def record_outcome(agent_name: str, outcome: str):
    with tables.open_file(str(RESULTS_FILE_PATH), "a") as f:
        vt = f.get_node("/agents", agent_name)
        for row in vt.iterrows(start=-1):
            if outcome == "WIN":
                row["won"] += 1
            elif outcome == "LOSS":
                row["lost"] += 1
            elif outcome == "DRAW":
                row["drawn"] += 1
            elif outcome == "FAIL":
                row["failed"] += 1
            row.update()
        vt.flush()


def add_game(game_result: GameResult):
    name_1 = game_result.result_1.name
    name_2 = game_result.result_2.name
    if "agent" not in name_1:
        add_game_for_agent(name_1, game_result)
    if "agent" not in name_2:
        add_game_for_agent(name_2, game_result)
    record_outcome(name_1, game_result.result_1.outcome)
    record_outcome(name_2, game_result.result_2.outcome)


def add_game_for_agent(agent_name: str, game_result: GameResult):
    fp = agent_games_file_path(agent_name)
    version = get_agent_version(agent_name)
    f = tables.open_file(str(fp), "w" if not fp.exists() else "a")
    ver_str = f"v{version:06}"
    try:
        vg = f.get_node("/", ver_str)
    except tables.NoSuchNodeError:
        vg = f.create_group("/", ver_str)
    try:
        gt = f.get_node(vg, "games")
    except tables.NoSuchNodeError:
        gt = f.create_table(vg, "games", GameOutcomeRow)

    result_1 = game_result.result_1
    result_2 = game_result.result_2
    key = gt.nrows
    moved_first = agent_name == result_1.name
    opponent_name = result_2.name if moved_first else result_1.name
    gt_row = gt.row
    gt_row["key"] = key
    gt_row["opponent"] = opponent_name
    gt_row["version"] = get_agent_version(opponent_name)
    gt_row["moved_first"] = moved_first
    gt_row["outcome"] = result_1.outcome if moved_first else result_2.outcome
    gt_row["when/time_str"] = game_result.time_str
    gt_row["when/time_sec"] = game_result.time_sec

    gt_row.append()
    gt.flush()

    key_str = f"g{key:06}"
    gg = f.create_group(vg, key_str)

    moves_1 = np.array(result_1.moves)
    moves_2 = np.array(result_2.moves)
    max_moves = max(moves_1.size, moves_2.size)
    moves = np.empty((max_moves, 2), dtype=np.int8)
    moves.fill(-1)
    moves[: moves_1.size, 0] = moves_1
    moves[: moves_2.size, 1] = moves_2

    ga = f.create_carray(gg, "moves", tables.Atom.from_dtype(moves.dtype), moves.shape)
    ga[...] = moves[...]

    mt_1 = np.array(result_1.move_times)
    mt_2 = np.array(result_2.move_times)
    max_mt = max(mt_1.size, mt_2.size)
    mts = np.empty((max_mt, 2), dtype=mt_1.dtype)
    mts.fill(-1)
    mts[: mt_1.size, 0] = mt_1
    mts[: mt_2.size, 1] = mt_2

    ga = f.create_carray(gg, "move_times", tables.Atom.from_dtype(mts.dtype), mts.shape)
    ga[...] = mts[...]


    if moved_first:
        stderr = result_1.stderr
        stdout = result_1.stdout
    else:
        stderr = result_2.stderr
        stdout = result_2.stdout

    f.create_array(gg, "stderr", stderr.encode("utf-8"))
    f.create_array(gg, "stdout", stdout.encode("utf-8"))

    f.flush()
    f.close()
    # agent_1, agent_2, outcome, moves_1, moves_2


def initialize(agent_names: Iterable[str]):
    if not RESULTS_FILE_PATH.exists():
        agent_names = list(agent_names)
        t_str = time.ctime()
        t_sec = time.time()
        with tables.open_file(str(RESULTS_FILE_PATH), "w") as f:
            ct = f.create_table("/", "current", CurrentAgentRow, createparents=True)
            for agent_name, row in zip(agent_names, ct.row):
                row["name"] = agent_name
                row["uploaded/time_str"] = t_str
                row["uploaded/time_sec"] = t_sec
                row.append()
            ct.flush()
        for agent_name in agent_names:
            add_agent(agent_name)


def agent_games_file_path(agent_name):
    return DATA_DIR / f"{agent_name}.h5"
