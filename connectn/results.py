from pathlib import Path
import time
import tables
import numpy as np
from tables import IsDescription, Int32Col, StringCol, Float64Col, BoolCol
from connectn.utils import DATA_DIR
from connectn.game import GameResult
from typing import Iterable, List, Dict, Any, TypeVar

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


def initialize(agent_names: Iterable[str]) -> None:
    if not RESULTS_FILE_PATH.exists():
        if not DATA_DIR.exists():
            DATA_DIR.mkdir()
        agent_names = list(agent_names)
        t_str = time.ctime()
        t_sec = time.time()
        with tables.open_file(str(RESULTS_FILE_PATH), "w") as results_file:
            ct = results_file.create_table(
                "/", "current", CurrentAgentRow, createparents=True
            )
            for agent_name, row in zip(agent_names, ct.row):
                row["name"] = agent_name
                row["uploaded/time_str"] = t_str
                row["uploaded/time_sec"] = t_sec
                row.append()
            results_file.flush()
        for agent_name in agent_names:
            add_agent(agent_name)


def add_agent(agent_name: str) -> None:
    with tables.open_file(str(RESULTS_FILE_PATH), "a") as results_file:
        __add_agent(results_file, agent_name)
        results_file.flush()


def add_game(game_result: GameResult) -> None:
    name_1 = game_result.result_1.name
    name_2 = game_result.result_2.name

    with tables.open_file(str(RESULTS_FILE_PATH), "a") as results_file:
        __check_for_agent(results_file, name_1)
        __check_for_agent(results_file, name_2)

        if __record_games_for_agent(name_1):
            agent_version = __get_current_agent_version(results_file, name_1)
            __add_game_for_agent(name_1, agent_version, game_result)
        if __record_games_for_agent(name_2):
            agent_version = __get_current_agent_version(results_file, name_2)
            __add_game_for_agent(name_2, agent_version, game_result)

        __record_outcome(results_file, game_result)


def get_current_agent_version(agent_name: str) -> int:
    with tables.open_file(str(RESULTS_FILE_PATH), "r") as results_file:
        version = __get_current_agent_version(results_file, agent_name)
    return version


def get_agent_version_numbers(agent_name: str) -> List[int]:
    versions = []
    with tables.open_file(str(RESULTS_FILE_PATH), "r") as results_file:
        av_table = results_file.get_node("/agents", agent_name)
        for row in av_table:
            versions.append(row["version"])
    return versions


def get_game_numbers_for_agent_version(
    agent_name: str, agent_version: int
) -> List[int]:
    game_number = []
    fp = __agent_games_file_path(agent_name)
    if fp.exists():
        with tables.open_file(str(fp), "r") as agent_file:
            try:
                vg = agent_file.get_node("/", __version_string(agent_version))
                gt = agent_file.get_node(vg, "games")
                for row in gt:
                    game_number.append(row["game_number"])
            except tables.NoSuchNodeError:
                print(
                    f"No record of games for {agent_name} for version {agent_version}"
                )
    return game_number


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

    fp = __agent_games_file_path(agent_name)
    with tables.open_file(str(fp), "r") as agent_file:
        vg = agent_file.get_node("/", __version_string(agent_version))
        gt = agent_file.get_node(vg, "games")

        found = False
        for gt_row in gt.where(f"(game_number == {game_number})"):
            assert (
                not found
            ), f"Found two entries for {agent_name} version {agent_version}, game number {game_number}"
            for k in game_result_cols:
                value = gt_row[k]
                game_info[k] = value.decode() if isinstance(value, bytes) else value
            found = True

        gg = agent_file.get_node(vg, __game_string(game_number))
        game_info["moves"] = gg.moves.read()
        game_info["move_times"] = gg.move_times.read()
        game_info["stdout"] = []
        game_info["stderr"] = []

        for stdout in gg.stdout:
            game_info["stdout"].append(stdout.decode())
        for stderr in gg.stderr:
            game_info["stderr"].append(stderr.decode())

    return game_info


def __check_for_agent(results_file: tables.File, agent_name: str) -> None:
    found = False
    for _ in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert not found, "Somehow there was more than one row with the same agent name"
        found = True
    if not found:
        __add_agent(results_file, agent_name)


def __record_games_for_agent(agent_name: str) -> bool:
    return "agent" not in agent_name


def __add_game_for_agent(
    agent_name: str, agent_version: int, game_result: GameResult
) -> None:

    fp = __agent_games_file_path(agent_name)

    with tables.open_file(str(fp), "a") as agent_file:
        ver_str = __version_string(agent_version)
        try:
            vg = agent_file.get_node("/", ver_str)
        except tables.NoSuchNodeError:
            vg = agent_file.create_group("/", ver_str)
        try:
            gt = agent_file.get_node(vg, "games")
        except tables.NoSuchNodeError:
            gt = agent_file.create_table(vg, "games", GameOutcomeRow)

        result_1 = game_result.result_1
        result_2 = game_result.result_2
        game_number = gt.nrows
        moved_first = agent_name == result_1.name
        opponent_name = result_2.name if moved_first else result_1.name
        gt_row = gt.row
        gt_row["game_number"] = game_number
        gt_row["opponent"] = opponent_name
        gt_row["version"] = get_current_agent_version(opponent_name)
        gt_row["moved_first"] = moved_first
        gt_row["outcome"] = result_1.outcome if moved_first else result_2.outcome
        gt_row["when/time_str"] = game_result.time_str
        gt_row["when/time_sec"] = game_result.time_sec

        gt_row.append()

        game_number_str = __game_string(game_number)
        gg = agent_file.create_group(vg, game_number_str)

        __add_array(agent_file, gg, "moves", result_1.moves, result_2.moves)
        __add_array(
            agent_file, gg, "move_times", result_1.move_times, result_2.move_times
        )
        __add_vlarray(
            agent_file,
            gg,
            "stderr",
            result_1.stderr if moved_first else result_2.stderr,
        )
        __add_vlarray(
            agent_file,
            gg,
            "stdout",
            result_1.stdout if moved_first else result_2.stdout,
        )

        agent_file.flush()


def __add_agent(results_file: tables.File, agent_name: str) -> None:
    try:
        av_table = results_file.get_node("/agents", agent_name)
    except tables.NoSuchNodeError:
        av_table = results_file.create_table(
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
    for ac_row in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert not found, "Somehow there was more than one row with the same agent name"
        found = True
        ac_row["version"] = n_versions
        ac_row["uploaded/time_str"] = t_str
        ac_row["uploaded/time_sec"] = t_uploaded
        ac_row.update()

    if not found:
        ac_row = results_file.root.current.row
        ac_row["name"] = agent_name
        ac_row["version"] = n_versions
        ac_row["uploaded/time_str"] = t_str
        ac_row["uploaded/time_sec"] = t_uploaded
        ac_row.append()

    results_file.root.current.flush()


def __get_current_agent_version(results_file: tables.File, agent_name: str) -> int:
    found = False
    version = -1
    for ac_row in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert not found, "Somehow there was more than one row with the same agent name"
        found = True
        version = ac_row["version"]
    return version


def __record_outcome(results_file: tables.File, game_result: GameResult) -> None:
    for result in (game_result.result_1, game_result.result_2):
        agent_name = result.name
        outcome = result.outcome
        vt = results_file.get_node("/agents", agent_name)
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


ListType = List[TypeVar("T")]  # Assures that both l1 and l2 contain the same type T


def __add_array(
    file: tables.File, where: tables.Group, name: str, l1: ListType, l2: ListType
) -> None:
    a1 = np.array(l1)
    a2 = np.array(l2)
    a12 = np.empty((max(a1.size, a2.size), 2), dtype=a1.dtype)
    a12.fill(-1)
    a12[: a1.size, 0] = a1
    a12[: a2.size, 1] = a2

    ca = file.create_carray(
        where,
        name,
        tables.Atom.from_dtype(a12.dtype),
        a12.shape,
        filters=compression_filter,
    )
    ca[...] = a12[...]


def __add_vlarray(
    file: tables.File, where: tables.group, name: str, ls: List[str]
) -> None:
    from tables import VLStringAtom

    atom: tables.Atom = VLStringAtom()

    vla = file.create_vlarray(
        where, name, atom=atom, filters=compression_filter, expectedrows=len(ls),
    )
    for s in ls:
        vla.append(s.encode("utf-8"))


def __agent_games_file_path(agent_name) -> Path:
    return DATA_DIR / f"{agent_name}.h5"


def __version_string(agent_version: int) -> str:
    return f"v{agent_version:06}"


def __game_string(game_number: int) -> str:
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
