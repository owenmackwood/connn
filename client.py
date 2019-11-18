import os
import tempfile
import tarfile
import socket
import importlib
from connectn.utils import ComfyStockings, LISTEN_PORT, ARCHIVE_FORMAT

EXCLUDE_FILES = [
    "__pycache__",
    ".idea",
    ".git",
    ".gitignore",
    ".metadata",
    ".pyc",
    ".pytest_cache",
]

# group_name, password = 'group_a', 'Ixdk5FOn'
group_name, password = "group_b", "jevfR7Fc"
# group_name, password = 'group_c', 'X7Xdm0NV'
# module_location = '.'.join(('connectn.agents', group_name))
# module_location = 'connectn.agents'.agent_random'
module_location = "connectn.agents.agent_rows"
module = importlib.import_module(module_location)
module_path = module.__path__[0]


def exclude_files(tarinfo):
    for ef in EXCLUDE_FILES:
        if ef == tarinfo.name or tarinfo.name.endswith(ef):
            return None
    return tarinfo


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cs:
    cs.connect(("localhost", LISTEN_PORT))
    with ComfyStockings(cs) as scs:
        scs.handshake_wait()

        print("Sending authentication request.")
        scs.write(f"{group_name},{password}")
        msg = scs.read_wait()
        print(f"Authentication: {msg}")

        if "OK" == msg:
            base_name = os.path.join(
                tempfile.gettempdir(), module_location.split(".")[-1]
            )
            file_path = ".".join((base_name, ARCHIVE_FORMAT))
            with tarfile.open(file_path, "w") as tar:
                for fn in os.listdir(module_path):
                    tar.add(os.path.join(module_path, fn), fn, filter=exclude_files)

            fs = os.path.getsize(file_path)
            print(f"Bytes to send: {fs}")
            scs.write(str(fs))
            with open(file_path, "rb") as f:
                scs.write(f.read())
            os.remove(file_path)

            msg = scs.read_wait()
            print(f"Agent upload: {msg}")
