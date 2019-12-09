import Stockings
import time
from time import sleep
from typing import Callable, Tuple, Optional, List
import numpy as np
import os
import sys
from pathlib import Path

LISTEN_PORT = 2323
ARCHIVE_FORMAT = "tar"
DATA_DIR: Path = Path.home() / "ppp_results"
KEY_SALT_FILE: Path = DATA_DIR / "keys_salts"
MOVE_TIME_MAX = 20.0
STATE_MEMORY_MAX = 2 ** 30  # Max of 1 GB
ON_CLUSTER = False


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
        while self.active and not self.handshakeComplete:
            print("Waiting for handshake")
            sleep(0.1)
        if not self.active:
            raise InactiveSocket("Socket became inactive while waiting for handshake")
        print(f"Handshake complete")

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

    for agent_name, archive_path in updated_agent_archives:
        module_path = os.path.join(cna.__path__[0], agent_name)
        if os.path.exists(module_path):
            shutil.rmtree(module_path)
        os.makedirs(module_path)
        shutil.unpack_archive(archive_path, module_path, ARCHIVE_FORMAT)
        os.remove(archive_path)
        results.add_agent(agent_name)
    updated_agents = [agent_name for agent_name, _ in updated_agent_archives]
    return updated_agents
