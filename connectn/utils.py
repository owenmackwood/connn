import Stockings
import time
from time import sleep
from typing import Callable, Tuple, Optional, List
import numpy as np
import os
import sys
from pathlib import Path
import platform
import logging

LISTEN_PORT = 2323
PROTOCOL_VERSION = 1
ARCHIVE_FORMAT = "tar"
ROOT_DATA_DIR: Path = Path.home() / "tournament"
SERVER_PROCESS_LOG: Path = ROOT_DATA_DIR / "server.log"
GAME_PROCESS_LOG: Path = ROOT_DATA_DIR / "game.log"
SERVER_PROCESS_DATA_DIR: Path = ROOT_DATA_DIR / "server_process"
KEY_SALT_FILE: Path = SERVER_PROCESS_DATA_DIR / "keys_salts"
GAME_PROCESS_DATA_DIR: Path = ROOT_DATA_DIR / "game_process"
TEMP_DIR: Path = GAME_PROCESS_DATA_DIR / "tmp"
MOVE_TIME_MAX = 10.0
STATE_MEMORY_MAX = 2 ** 30  # Max of 1 GB
RUN_ALL_EVERY = 6  # Only rerun every six hours
ON_CLUSTER = platform.node() == "cluster"
LOG_LEVEL = "INFO"
PLAY_ALL = False
TOURNAMENT_FILE = "all_games"
AGENTS_PER_USER = 2


if not ROOT_DATA_DIR.exists():
    ROOT_DATA_DIR.mkdir()


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, int, Optional[SavedState]], Tuple[int, Optional[SavedState]]
]


class InactiveSocket(Exception):
    pass


class ComfyStockings(Stockings.Stocking):
    def __exit__(self, typ, value, tb):
        super().__exit__(typ, value, tb)
        self.sock.close()

    def handshake_wait(self):
        import logging

        logger = logging.getLogger(__name__)

        while self.active and not self.handshakeComplete:
            logger.info("Waiting for handshake")
            sleep(0.1)
        if not self.active:
            raise InactiveSocket("Socket became inactive while waiting for handshake")
        logger.info(f"Handshake complete")

    def read_wait(self, delay=0.1, timeout=10.0):
        t0 = time.time()
        while self.active:
            msg = self.read()
            if msg is not None:
                return msg
            sleep(delay)
            dt = time.time() - t0
            if dt > timeout:
                raise InactiveSocket("Timed out waiting for reply")
        raise InactiveSocket("Socket became inactive while waiting for reply")


def parse_arguments():
    import argparse

    global LOG_LEVEL, MOVE_TIME_MAX, SERVER_PROCESS_LOG, STATE_MEMORY_MAX, PLAY_ALL, RUN_ALL_EVERY
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--logfile",
        help="Path and name of log file.",
        type=str,
        default=f"{SERVER_PROCESS_LOG!s}",
    )
    parser.add_argument(
        "--level",
        help="Logging level.",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=LOG_LEVEL,
    )
    parser.add_argument(
        "--maxtime",
        help="Maximum time per move in seconds.",
        type=float,
        default=MOVE_TIME_MAX,
    )
    parser.add_argument(
        "--maxsize",
        help="Maximum data permitted for saved_state in GiB.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--playall",
        help="Play all possible matches on startup.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--schedule",
        help="Number of hours to wait before re-running tournament.",
        type=int,
        default=RUN_ALL_EVERY,
    )

    args = parser.parse_args()

    SERVER_PROCESS_LOG = Path(args.logfile).expanduser()
    LOG_LEVEL = args.level.upper()
    MOVE_TIME_MAX = args.maxtime
    STATE_MEMORY_MAX = int(np.round(STATE_MEMORY_MAX * args.maxsize))
    PLAY_ALL = args.playall
    RUN_ALL_EVERY = args.schedule


def configure_logging(log_file: Path, log_level: str=LOG_LEVEL):
    numeric_level = getattr(logging, log_level, None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        filename=f"{log_file!s}",
        filemode="w",
        level=numeric_level,
        format="%(asctime)s %(levelname)s/%(processName)s:%(name)s:%(message)s",
    )
    root = logging.getLogger()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(numeric_level)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s/%(processName)s:%(name)s:%(message)s")
    )
    root.addHandler(stdout_handler)


def get_size(obj, seen=None) -> int:
    """Recursively finds size of objects in bytes"""
    import inspect

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, "__dict__"):
        for cls in obj.__class__.__mro__:
            if "__dict__" in cls.__dict__:
                d = cls.__dict__["__dict__"]
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif isinstance(obj, np.ndarray) and obj.base is not None:
        # The check for ndarray and base is not None is whether obj is a view on an ndarray,
        # which needs to be iterated on, since getsizeof(view) does not reflect the
        size += sys.getsizeof(obj.base)
    elif hasattr(obj, "__iter__") and not isinstance(
        obj, (str, bytes, bytearray, np.ndarray)
    ):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
        size += sum(
            get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s)
        )

    return size


def mib(n_bytes: int) -> float:
    return n_bytes / 2 ** 20


def update_user_agent_code(updated_agent_archives: List[Tuple[str, str]]) -> List[str]:
    from connectn import results
    import connectn.agents as cna
    import shutil
    import logging

    logger = logging.getLogger(__name__)

    for agent_name, archive_path in updated_agent_archives:
        module_path = os.path.join(cna.__path__[0], agent_name)
        logger.info(f"Writing module files for {agent_name} to {module_path}")
        if os.path.exists(module_path):
            shutil.rmtree(module_path)
        os.makedirs(module_path)
        shutil.unpack_archive(archive_path, module_path, ARCHIVE_FORMAT)
        os.remove(archive_path)
        results.add_agent(agent_name)
    updated_agents = [agent_name for agent_name, _ in updated_agent_archives]
    return updated_agents
