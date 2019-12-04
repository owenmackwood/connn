from pathlib import Path
import time
import tables
import numpy as np
from tables import IsDescription, Int32Col, StringCol, Float64Col, BoolCol
from connectn.utils import DATA_DIR
from connectn.game import GameResult
from typing import Iterable, List, Dict, Any

RESULTS_FILE_PATH = DATA_DIR / "results.h5"
name_size = 32
compression_filter = tables.Filters(complevel=5, complib="zlib")

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
        games -> Date, Time, Opponent, Opponent version, Played first (bool), Outcome, Game number
        <game number>
            moves -> Player 1 move, Player 2 move
            stdout -> string array
            stderr -> string array
"""


class GameOutcomeRow(IsDescription):
    game_number = Int32Col(pos=0)
    opponent = StringCol(name_size, pos=1)
    version = Int32Col(pos=2)
    moved_first = BoolCol(pos=3)
    outcome = StringCol(5, pos=4)
    when = TimeStamp()


def add_agent(agent_name: str) -> None:
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


def get_agent_version(agent_name: str) -> int:
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


def check_for_agent(agent_name: str) -> None:
    found = False
    with tables.open_file(str(RESULTS_FILE_PATH), "r") as f:
        for _ in f.root.current.where(f'(name == b"{agent_name}")'):
            assert (
                not found
            ), "Somehow there was more than one row with the same agent name"
            found = True
    if not found:
        add_agent(agent_name)


def record_outcome(game_result: GameResult) -> None:
    with tables.open_file(str(RESULTS_FILE_PATH), "a") as f:
        for result in (game_result.result_1, game_result.result_2):
            agent_name = result.name
            outcome = result.outcome
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


def record_games_for_agent(agent_name: str) -> bool:
    return "agent" not in agent_name


def add_game(game_result: GameResult) -> None:
    name_1 = game_result.result_1.name
    name_2 = game_result.result_2.name

    check_for_agent(name_1)
    check_for_agent(name_2)

    if record_games_for_agent(name_1):
        add_game_for_agent(name_1, game_result)
    if record_games_for_agent(name_2):
        add_game_for_agent(name_2, game_result)

    record_outcome(game_result)


def add_game_for_agent(agent_name: str, game_result: GameResult) -> None:
    from tables import VLStringAtom

    fp = agent_games_file_path(agent_name)
    agent_version = get_agent_version(agent_name)

    with tables.open_file(str(fp), "w" if not fp.exists() else "a") as f:
        ver_str = version_string(agent_version)
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
        game_number = gt.nrows
        moved_first = agent_name == result_1.name
        opponent_name = result_2.name if moved_first else result_1.name
        gt_row = gt.row
        gt_row["game_number"] = game_number
        gt_row["opponent"] = opponent_name
        gt_row["version"] = get_agent_version(opponent_name)
        gt_row["moved_first"] = moved_first
        gt_row["outcome"] = result_1.outcome if moved_first else result_2.outcome
        gt_row["when/time_str"] = game_result.time_str
        gt_row["when/time_sec"] = game_result.time_sec

        gt_row.append()
        gt.flush()

        game_number_str = game_string(game_number)
        gg = f.create_group(vg, game_number_str)

        moves_1 = np.array(result_1.moves)
        moves_2 = np.array(result_2.moves)
        max_moves = max(moves_1.size, moves_2.size)
        moves = np.empty((max_moves, 2), dtype=np.int8)
        moves.fill(-1)
        moves[: moves_1.size, 0] = moves_1
        moves[: moves_2.size, 1] = moves_2

        ga = f.create_carray(
            gg,
            "moves",
            tables.Atom.from_dtype(moves.dtype),
            moves.shape,
            filters=compression_filter,
        )
        ga[...] = moves[...]

        mt_1 = np.array(result_1.move_times)
        mt_2 = np.array(result_2.move_times)
        max_mt = max(mt_1.size, mt_2.size)
        mts = np.empty((max_mt, 2), dtype=mt_1.dtype)
        mts.fill(-1)
        mts[: mt_1.size, 0] = mt_1
        mts[: mt_2.size, 1] = mt_2

        ga = f.create_carray(
            gg,
            "move_times",
            tables.Atom.from_dtype(mts.dtype),
            mts.shape,
            filters=compression_filter,
        )
        ga[...] = mts[...]

        if moved_first:
            stderr: List[str] = result_1.stderr
            stdout: List[str] = result_1.stdout
        else:
            stderr: List[str] = result_2.stderr
            stdout: List[str] = result_2.stdout

        vla = f.create_vlarray(
            gg,
            "stderr",
            atom=VLStringAtom(),
            filters=compression_filter,
            expectedrows=len(stderr),
        )
        for s in stderr:
            vla.append(s.encode("utf-8"))
        vla = f.create_vlarray(
            gg,
            "stdout",
            atom=VLStringAtom(),
            filters=compression_filter,
            expectedrows=len(stdout),
        )
        for s in stdout:
            vla.append(s.encode("utf-8"))

        f.flush()


def get_game_for_agent(
    agent_name: str, agent_version: int, game_number: int
) -> Dict[str, Any]:
    game_result_cols = [
        "opponent",
        "version",
        "moved_first",
        "outcome",
        "when/time_str",
        "when/time_sec",
    ]
    game_info = {col: None for col in game_result_cols}

    fp = agent_games_file_path(agent_name)
    with tables.open_file(str(fp), "r") as f:
        vg = f.get_node("/", version_string(agent_version))
        gt = f.get_node(vg, "games")

        found = False
        for gt_row in gt.where(f"(game_number == {game_number})"):
            assert (
                not found
            ), f"Found two entries for {agent_name} version {agent_version}, game number {game_number}"
            for k in game_result_cols:
                value = gt_row[k]
                game_info[k] = value.decode() if isinstance(value, bytes) else value
            found = True

        gg = f.get_node(vg, game_string(game_number))
        game_info["moves"] = gg.moves.read()
        game_info["move_times"] = gg.move_times.read()
        game_info["stdout"] = []
        game_info["stderr"] = []

        for stdout in gg.stdout:
            game_info["stdout"].append(stdout.decode())
        for stderr in gg.stderr:
            game_info["stderr"].append(stderr.decode())

    return game_info


def get_agent_version_numbers(agent_name: str) -> List[int]:
    versions = []
    with tables.open_file(str(RESULTS_FILE_PATH), "r") as f:
        av_table = f.get_node("/agents", agent_name)
        for row in av_table:
            versions.append(row["version"])
    return versions


def get_game_numbers_for_agent_version(
    agent_name: str, agent_version: int
) -> List[int]:
    game_number = []
    fp = agent_games_file_path(agent_name)
    if fp.exists():
        with tables.open_file(str(fp), "r") as f:
            try:
                vg = f.get_node("/", version_string(agent_version))
                gt = f.get_node(vg, "games")
                for row in gt:
                    game_number.append(row["game_number"])
            except tables.NoSuchNodeError:
                print(
                    f"No record of games for {agent_name} for version {agent_version}"
                )
    return game_number


def initialize(agent_names: Iterable[str]) -> None:
    if not RESULTS_FILE_PATH.exists():
        if not DATA_DIR.exists():
            DATA_DIR.mkdir()
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


def agent_games_file_path(agent_name) -> Path:
    return DATA_DIR / f"{agent_name}.h5"


def version_string(agent_version: int) -> str:
    return f"v{agent_version:06}"


def game_string(game_number: int) -> str:
    return f"g{game_number:06}"


if __name__ == "__main__":
    from connectn.users import agents

    for agent_name in agents():
        agent_versions = get_agent_version_numbers(agent_name)
        for av in agent_versions:
            print(f"{agent_name} version: {av}")
            game_number = get_game_numbers_for_agent_version(agent_name, av)
            for gn in game_number:
                print(f"{agent_name} version {av}, game number: {gn}")
                print(get_game_for_agent(agent_name, av, gn))
        if not agent_versions:
            print(f"No agent versions for {agent_name}")
