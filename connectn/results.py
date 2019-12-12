from pathlib import Path
import time
import tables
import numpy as np
import logging
from tables import IsDescription, Int32Col, StringCol, Float64Col, BoolCol
from connectn.utils import GAME_PROCESS_DATA_DIR
from connectn.game import GameResult
from typing import Iterable, List, Dict, Any, TypeVar, Union, Optional


RESULTS_FILE_PATH = GAME_PROCESS_DATA_DIR / "results.h5"
name_size = 32
outcome_size = 16
compression_filter = tables.Filters(complevel=5, complib="zlib")
logger = logging.getLogger(__name__)
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


class AgentGameOutcomeRow(IsDescription):
    game_number = Int32Col(pos=0)
    opponent = StringCol(name_size, pos=1)
    version = Int32Col(pos=2)
    moved_first = BoolCol(pos=3)
    outcome = StringCol(outcome_size, pos=4)
    when = TimeStamp()


class GameSummaryAgent(IsDescription):
    name = StringCol(name_size, pos=0)
    version = Int32Col(pos=1)
    rating = Float64Col(pos=2, dflt=0.0)
    outcome = StringCol(outcome_size, pos=3)
    time_med = Float64Col(pos=4)
    time_max = Float64Col(pos=5)
    state_size_med = Int32Col(pos=6)
    state_size_max = Int32Col(pos=6)


class GameSummaryRow(IsDescription):
    agent1 = GameSummaryAgent()
    agent2 = GameSummaryAgent()
    when = TimeStamp()
    winner = Int32Col()


def initialize(agent_names: Iterable[str]) -> None:
    if not RESULTS_FILE_PATH.exists():
        agent_names = list(agent_names)
        t_str = time.ctime()
        t_sec = time.time()
        with tables.open_file(f"{RESULTS_FILE_PATH!s}", "w") as results_file:
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


def add_agent(agent_name: str, file_path: Path = RESULTS_FILE_PATH) -> None:
    """
    Records a new version for agent `agent_name` by adding a row to the
    `/current` table, adding a new sub-group at `/agent_name/v<version_number>`
    where games played by that version of the agent are stored.

    If the agent does not exist yet, it is added to the results file.

    Parameters
    ----------
    agent_name : str
        The name of the agent
    file_path : Path, optional
    """
    with tables.open_file(f"{file_path!s}", "a") as results_file:
        _add_agent(results_file, agent_name)
        results_file.flush()


def add_game(game_result: GameResult, file_path: Path = RESULTS_FILE_PATH) -> None:
    """
    Records the result of a game played between two agents.
    This includes updating the win/loss/draw count for the current agent
    version, as well as storing a complete record of moves made, time per
    move, stdout and stderr, and other information useful for diagnosing
    and evaluating agent behaviour.

    Parameters
    ----------
    game_result : GameResult
        Container object for all game data
    file_path : Path, optional
    """
    agent_1_name = game_result.result_1.name
    agent_2_name = game_result.result_2.name

    with tables.open_file(f"{file_path!s}", "a") as results_file:
        _check_for_agent(results_file, agent_1_name)
        _check_for_agent(results_file, agent_2_name)

        agent_1_version = _get_current_agent_version(results_file, agent_1_name)
        agent_2_version = _get_current_agent_version(results_file, agent_2_name)
        if record_games_for_agent(agent_1_name):
            with tables.open_file(
                f"{agent_games_file_path(agent_1_name)!s}", "a"
            ) as agent_file:
                _add_game_for_agent(
                    agent_file,
                    agent_1_name,
                    agent_1_version,
                    agent_2_version,
                    game_result,
                )
        if record_games_for_agent(agent_2_name):
            with tables.open_file(
                f"{agent_games_file_path(agent_2_name)!s}", "a"
            ) as agent_file:
                _add_game_for_agent(
                    agent_file,
                    agent_2_name,
                    agent_2_version,
                    agent_1_version,
                    game_result,
                )

        _record_outcome(results_file, game_result, agent_1_version, agent_2_version)

        results_file.flush()


def get_current_agent_version(
    agent_name: str, file_path: Path = RESULTS_FILE_PATH
) -> int:
    """
    Retrieve the current version number of agent `agent_name`

    Parameters
    ----------
    agent_name : str
        Name of the agent being queried
    file_path : Path, optional

    Returns
    -------
    version : int
        The current version of agent `agent_name`
    """
    with tables.open_file(f"{file_path!s}", "r") as results_file:
        version = _get_current_agent_version(results_file, agent_name)
    return version


def get_agent_version_numbers(
    agent_name: str, file_path: Path = RESULTS_FILE_PATH
) -> List[int]:
    """
    Retrieve all version numbers for agent `agent_name`
    Versions are stored in the results file in the table `/agents/<agent_name>`

    Parameters
    ----------
    agent_name : str
        Name of the agent being queried
    file_path : Path, optional

    Returns
    -------
    versions : list of int
    """
    versions = []
    with tables.open_file(f"{file_path!s}", "r") as results_file:
        avt: tables.Table = results_file.get_node("/agents", agent_name)
        versions += avt.col("version").tolist()
    return versions


def get_game_numbers_for_agent_version(
    agent_name: str, agent_version: int, agent_file_path: Optional[Path] = None
) -> List[int]:
    """
    Retrieve the identifying number for every game played by
    a particular version of an agent. Those game numbers can be
    used to retrieve a complete record of each game by calling
    `get_game_for_agent`.


    Parameters
    ----------
    agent_name : str
        Name of the agent being queried
    agent_version :
        The version of the agent being queried
    agent_file_path : Path, optional

    Returns
    -------
    game_numbers : list of int
    """
    game_numbers = []
    fp = (
        agent_games_file_path(agent_name)
        if agent_file_path is None
        else agent_file_path
    )
    if fp.exists():
        with tables.open_file(f"{fp!s}", "r") as agent_file:
            try:
                vg: tables.Group = agent_file.get_node(
                    "/", _version_string(agent_version)
                )
                gt: tables.Table = agent_file.get_node(vg, "games")
                game_numbers += gt.col("game_number").tolist()
            except tables.NoSuchNodeError:
                logger.info(
                    f"No record of games for {agent_name} for version {agent_version}"
                )
    return game_numbers


def get_game_for_agent(
    agent_name: str,
    agent_version: int,
    game_number: int,
    agent_file_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Retrieve the complete record of a single game.
    Given the agent name, the agent version and the game number, this
    function loads the entire record of the game and returns it.

    Parameters
    ----------
    agent_name : str
        Name of the agent being queried
    agent_version : int
        The version of the agent being queried
    game_number : int
        The number identifying the game being queried
    agent_file_path : Path, optional

    Returns
    -------
    game_info : dict
        This dictionary contains the following keys:
                game_number : int
                opponent : str
                version : int
                moved_first : bool
                outcome : str
                when : tuple of str, float
                moves: ndarray of PlayerAction
                move_times: ndarray of float
                state_size: ndarray of int
                seeds: ndarray of int
                stdout: list of str
                stderr: list of str
    """
    game_info = {}
    fp = (
        agent_games_file_path(agent_name)
        if agent_file_path is None
        else agent_file_path
    )
    with tables.open_file(f"{fp!s}", "r") as agent_file:
        vg = agent_file.get_node("/", _version_string(agent_version))
        gt: tables.Table = agent_file.get_node(vg, "games")
        game_result_cols = gt.colnames

        def decode(v):
            if isinstance(v, tuple):
                return tuple(decode(vv) for vv in v)
            elif isinstance(v, bytes):
                return v.decode()
            else:
                return v

        found = False
        for gt_row in gt.where(f"(game_number == {game_number})"):
            msg = f"Found two entries for {agent_name} version {agent_version}, game number {game_number}"
            assert not found, msg
            found = True
            for k in game_result_cols:
                game_info[k] = decode(gt_row[k])

        if not found:
            msg = f"Game {game_number} for {agent_name} version {agent_version} was not found in the games table."
            logger.error(msg)

        try:
            gg: tables.Group = agent_file.get_node(vg, _game_string(game_number))
        except tables.NoSuchNodeError:
            msg = f"No complete record of game {game_number} for {agent_name} version {agent_version} exists."
            logger.error(msg)
        else:
            for n in ("moves", "move_times", "state_size", "seeds"):
                try:
                    game_info[n] = agent_file.get_node(gg, n).read()
                except tables.NoSuchNodeError:
                    msg = f"{n} not found for {agent_name}, version {agent_version}, game {game_number}"
                    logger.error(msg)

            for n in ("stdout", "stderr"):
                game_info[n] = []
                try:
                    for std_ in agent_file.get_node(gg, n):
                        game_info[n].append(std_.decode())
                except tables.NoSuchNodeError:
                    msg = f"{n} not found for {agent_name}, version {agent_version}, game {game_number}"
                    logger.error(msg)

    return game_info


AgentSummary = Dict[str, Union[str, int, float]]
GameSummary = Dict[str, Union[str, AgentSummary]]


def get_all_game_summaries(file_path: Path = RESULTS_FILE_PATH) -> List[GameSummary]:
    """
    Get a summary record of every game ever played.
    GameSummaries are dictionaries structured like so:
        `winner` : int
            Whether agent1, agent2 or neither won the game
        `when/time_str` : str
            When the game was played (in a human readable format)
        `when/time_sec` : float
            When the game was played (in seconds since the Epoch)
        `agent1` : AgentSummary
        `agent2` : AgentSummary
            Both AgentResults contain the following:
                `name` : str
                `version` : int
                `rating` : float
                `outcome` : str
                `time_med` : float
                `time_max` : float
                `state_size_med` : float
                `state_size_max` : float

    Parameters
    ----------
    file_path : Path, optional

    Returns
    -------
    all_games : list of GameSummary
    """
    time.time()
    all_games = []
    with tables.open_file(f"{file_path!s}", "r") as results_file:
        agt = results_file.root.all_games
        for gr in agt:
            game = {
                "winner": gr["winner"],
                "when/time_str": gr["when/time_str"].decode(),
                "when/time_sec": gr["when/time_sec"],
                "agent1": {},
                "agent2": {},
            }
            for i in (1, 2):
                agent_result = game[f"agent{i}"]
                for k in (
                    "name",
                    "version",
                    "rating",
                    "outcome",
                    "time_med",
                    "time_max",
                    "state_size_med",
                    "state_size_max",
                ):
                    val = gr[f"agent{i}/{k}"]
                    agent_result[k] = val.decode() if isinstance(val, bytes) else val
            all_games.append(game)
    return all_games


def _check_for_agent(results_file: tables.File, agent_name: str) -> None:
    """
    Check whether an agent exists in the `/current` table of the results
    file. If it does not, then add that agent using `_add_agent`

    Parameters
    ----------
    results_file : tables.File
    agent_name : str
    """
    found = False
    for _ in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert not found, "Somehow there was more than one row with the same agent name"
        found = True
    if not found:
        _add_agent(results_file, agent_name)


def record_games_for_agent(agent_name: str) -> bool:
    """
    Decide whether a complete record of a game should be
    recorded for the given agent. Currently complete records
    are not kept for any of the default agents, whose names
    always start with `agent`.

    Parameters
    ----------
    agent_name : str

    Returns
    -------
    bool
    """
    return not agent_name.startswith("agent")


def _add_game_for_agent(
    agent_file: tables.File,
    agent_name: str,
    agent_version: int,
    opponent_version: int,
    game_result: GameResult,
) -> None:
    """
    Create a complete record of a game played by an agent.

    Parameters
    ----------
    agent_file : tables.File
    agent_name : str
    agent_version : int
    opponent_version : int
    game_result : GameResult
    """

    ver_str = _version_string(agent_version)
    gt: tables.Table
    vg: tables.Group
    try:
        vg = agent_file.get_node("/", ver_str)
    except tables.NoSuchNodeError:
        vg = agent_file.create_group("/", ver_str)
    try:
        gt = agent_file.get_node(vg, "games")
    except tables.NoSuchNodeError:
        gt = agent_file.create_table(vg, "games", AgentGameOutcomeRow)

    result_1 = game_result.result_1
    result_2 = game_result.result_2
    game_number = gt.nrows
    moved_first = agent_name == result_1.name
    if moved_first:
        result_agent = result_1
        result_opponent = result_2
    else:
        result_agent = result_2
        result_opponent = result_1

    opponent_name = result_opponent.name
    gt_row = gt.row
    gt_row["game_number"] = game_number
    gt_row["opponent"] = opponent_name
    gt_row["version"] = opponent_version
    gt_row["moved_first"] = moved_first
    gt_row["outcome"] = result_agent.outcome
    gt_row["when/time_str"] = game_result.time_str
    gt_row["when/time_sec"] = game_result.time_sec

    gt_row.append()

    game_number_str = _game_string(game_number)
    gg = agent_file.create_group(vg, game_number_str)

    for n in ("moves", "move_times"):
        _add_array(agent_file, gg, n, getattr(result_1, n), getattr(result_2, n))

    _add_array(agent_file, gg, "state_size", result_agent.state_size)
    _add_array(agent_file, gg, "seeds", result_agent.seeds)

    for n in ("stdout", "stderr"):
        _add_vlarray(agent_file, gg, n, getattr(result_agent, n))

    agent_file.flush()


def _add_agent(results_file: tables.File, agent_name: str) -> None:
    """
    Implementation for `add_agent`. See that function for a description
    of this code.

    See Also
    --------
    add_agent : The externally callable wrapper for this function
    """
    avt: tables.Table
    try:
        avt = results_file.get_node("/agents", agent_name)
    except tables.NoSuchNodeError:
        avt = results_file.create_table(
            "/agents", agent_name, AgentVersionRow, createparents=True
        )

    n_versions = avt.nrows
    t_uploaded = time.time()
    t_str = time.ctime()
    av_row = avt.row
    av_row["version"] = n_versions
    av_row["uploaded/time_str"] = t_str
    av_row["uploaded/time_sec"] = t_uploaded
    av_row.append()
    avt.flush()

    found = False
    for ac_row in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert (
            not found
        ), f"There was more than one row with the same agent name: {agent_name}"
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


def _get_current_agent_version(results_file: tables.File, agent_name: str) -> int:
    """
    Retrieves the current version of an agent.

    Parameters
    ----------
    results_file : tables.File
    agent_name : str

    Returns
    -------
    version : int
    """
    found = False
    version = -1
    for ac_row in results_file.root.current.where(f'(name == b"{agent_name}")'):
        assert (
            not found
        ), f"There was more than one row with the same agent name: {agent_name}"
        found = True
        version = ac_row["version"]
    return version


def _record_outcome(
    results_file: tables.File, game_result: GameResult, *agent_version: int
) -> None:
    """
    Record the outcome of a single game for both agents that participated
    in it. More specifically, this function increments the number of
    wins / losses / draws / failures for the two agents.

    Parameters
    ----------
    results_file :
    game_result :
    """
    for result in (game_result.result_1, game_result.result_2):
        agent_name = result.name
        outcome = result.outcome
        vt: tables.Table = results_file.get_node("/agents", agent_name)
        first = True
        for row in vt.iterrows(start=-1):
            assert (
                first
            ), "We only want to update the last row, so this loop should only be entered once."
            first = False
            if outcome == "WIN":
                row["won"] += 1
            elif outcome == "LOSS":
                row["lost"] += 1
            elif outcome == "DRAW":
                row["drawn"] += 1
            else:
                row["failed"] += 1
            row.update()
        vt.flush()

    agt: tables.Table
    try:
        agt = results_file.root.all_games
    except tables.NoSuchNodeError:
        agt = results_file.create_table("/", "all_games", GameSummaryRow)
    gr = agt.row
    gr["winner"] = game_result.winner
    gr["when/time_str"] = game_result.time_str
    gr["when/time_sec"] = game_result.time_sec
    for i, result in enumerate((game_result.result_1, game_result.result_2), 1):
        gr[f"agent{i}/name"] = result.name
        gr[f"agent{i}/version"] = agent_version[i - 1]
        gr[f"agent{i}/rating"] = 0.0
        gr[f"agent{i}/outcome"] = result.outcome
        if result.move_times:
            gr[f"agent{i}/time_med"] = np.median(result.move_times)
            gr[f"agent{i}/time_max"] = np.max(result.move_times)
        else:
            gr[f"agent{i}/time_med"] = -1
            gr[f"agent{i}/time_max"] = -1
        if result.state_size:
            gr[f"agent{i}/state_size_med"] = np.median(result.state_size)
            gr[f"agent{i}/state_size_max"] = np.max(result.state_size)
        else:
            gr[f"agent{i}/state_size_med"] = -1
            gr[f"agent{i}/state_size_max"] = -1
    gr.append()
    agt.flush()


T = TypeVar("T", np.floating, np.integer, np.bool, int, float)
HomList = List[T]  # A homogeneous list, where all elements are the same type T


def _add_array(
    file: tables.File, where: tables.Group, name: str, *lists: HomList
) -> None:
    """
    Adds an homogeneous array to a tables file, where the array is filled
    with the contents of a set of lists. Each list in *lists is copied into
    a column of the resulting tables.CArray using the same ordering as *lists.

    Parameters
    ----------
    file : tables.File
    where : tables.Group
    name : str
    lists : list of lists, each containing the same scalar data type (e.g. float, int)
    """
    arrays = [np.array(ll) for ll in lists]
    nda = np.empty(
        (max(1, max(a.size for a in arrays)), len(arrays)), dtype=arrays[0].dtype
    )
    nda.fill(-1)
    for i, a in enumerate(arrays):
        nda[: a.size, i] = a

    ca: tables.CArray = file.create_carray(
        where,
        name,
        tables.Atom.from_dtype(nda.dtype),
        nda.shape,
        filters=compression_filter,
    )
    ca[...] = nda[...]


def _add_vlarray(
    file: tables.File,
    where: tables.group,
    name: str,
    to_store: List[Union[str, HomList]],
) -> None:
    """
    Adds a ragged array to a tables file. Each row in the array is
    populated by an element of `to_store`, which can contain either strings or
    lists of any of the types supported by PyTables. This includes
    floats, integers and other scalar data types.

    Parameters
    ----------
    file : tables.File
    where : tables.Group
    name : str
    to_store : list of either str or lists of scalars
    """
    if to_store:
        if isinstance(to_store[0], str):
            to_store = [s.encode("utf-8") for s in to_store]
            atom = tables.VLStringAtom()
        else:
            to_store = [np.array(ll) for ll in to_store]
            atom = tables.Atom.from_dtype(to_store[0].dtype)
    else:
        atom = tables.StringAtom(itemsize=1)

    vla = file.create_vlarray(
        where, name, atom=atom, filters=compression_filter, expectedrows=len(to_store),
    )
    for s in to_store:
        vla.append(s)


def agent_games_file_path(agent_name) -> Path:
    """ Path to the file containing complete game records for an agent.

    Parameters
    ----------
    agent_name : str

    Returns
    -------
    Path
    """
    return GAME_PROCESS_DATA_DIR / f"{agent_name}.h5"


def _version_string(agent_version: int) -> str:
    """ Convenience function used to produce standard
    string formatting for a given agent version.

    Parameters
    ----------
    agent_version : int

    Returns
    -------
    str
    """
    return f"v{agent_version:06}"


def _game_string(game_number: int) -> str:
    """ Convenience function used to produce standard
    string formatting for a given game number.

    Parameters
    ----------
    game_number : int

    Returns
    -------
    str
    """
    return f"g{game_number:06}"


if __name__ == "__main__":
    from connectn.users import agents

    show_summary = True
    if show_summary:
        # Example of how to get the summary record of all games
        path = Path("~/tournament/all_games-2019-12-12-14h02m18s.h5").expanduser()
        games = get_all_game_summaries(path)
        for g in games:
            for j in (1, 2):
                res = g[f"agent{j}"]
                if res["outcome"] not in ("WIN", "LOSS", "DRAW"):
                    print(
                        f"{g['when/time_str']} ({g['when/time_sec']} seconds since the Epoch)"
                    )
                    print(res["name"], res["outcome"])

    show_details = False
    if show_details:
        # Example of how to retrieve detailed records of games
        for an in agents():
            agent_versions = get_agent_version_numbers(an)
            for av in agent_versions:
                print(f"{an} version: {av}")
                for gn in get_game_numbers_for_agent_version(an, av):
                    print(f"{an} version {av}, game number: {gn}")
                    print(get_game_for_agent(an, av, gn))
            if not agent_versions:
                print(f"No agent versions for {an}")
