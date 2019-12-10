import argparse
from pathlib import Path

group_name = "group_a"
password = "rgIWtVRp"
module_location = Path("~/codebase/connn/connectn/agents/agent_mcts")
results_location = Path.home() / "tournament" / "results"

def parse_arguments():
    global group_name, password, module_location
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--username", help="Your group name.", type=str, default=group_name
    )
    parser.add_argument(
        "-p", "--password", help="The associated password.", type=str, default=password
    )
    parser.add_argument(
        "-a",
        "--agent",
        help="Path to your agent's Python package, e.g. `~/myproject/mycode/mypackage/`",
        type=str,
        default=f"{module_location!s}",
    )
    args = parser.parse_args()

    group_name = args.username
    password = args.password
    module_location = Path(args.agent).expanduser()


def exclude_files(tarinfo):
    excluded_files = [
        "__pycache__",
        ".idea",
        ".git",
        ".gitignore",
        ".metadata",
        ".pyc",
        ".pytest_cache",
    ]
    for ef in excluded_files:
        if ef == tarinfo.name or tarinfo.name.endswith(ef):
            return None
    return tarinfo


def connect():
    import os
    import tempfile
    import tarfile
    import socket
    import time
    from connectn.utils import ComfyStockings, LISTEN_PORT

    up_or_down = ""
    while not up_or_down:
        inp = input("Upload agent or download results? u / [d] ")
        if inp == "u":
            up_or_down = "UPLOAD"
        elif inp in ("d", ""):
            up_or_down = "DOWNLOAD"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cs:
        cs.connect(("localhost", LISTEN_PORT))
        with ComfyStockings(cs) as scs:
            scs.handshake_wait()

            print("Sending authentication request.")
            scs.write(f"{group_name},{password}")
            msg = scs.read_wait()

            if "OK" != msg:
                raise Exception(f"Authentication: {msg}")

            scs.write(up_or_down)
            if up_or_down == "UPLOAD":
                msg = scs.read_wait()
                if msg != "READY":
                    raise Exception(f"When waiting to upload server responded with: {msg}")

                with tempfile.NamedTemporaryFile() as tar_file:
                    with tarfile.open(tar_file.name, "w") as tar:
                        for fn in os.listdir(module_location):
                            tar.add(module_location / fn, fn, filter=exclude_files)
                    fs = f"{os.path.getsize(tar_file.name)!s}"
                    print(f"Bytes to send: {fs}")
                    scs.write(fs)
                    with open(tar_file.name, "rb") as f:
                        scs.write(f.read())

                msg = scs.read_wait()
                print(f"Agent upload: {msg}")
            else:
                bytes_expected = int(scs.read_wait())
                print(f"Now starting transfer of {bytes_expected} bytes")
                data = scs.read_wait()
                bytes_received = len(data)
                print(f"Received bytes {bytes_received}, expected {bytes_expected}")
                msg = "FAIL"
                if not results_location.exists():
                    results_location.mkdir()

                file_path = (results_location / time.strftime('%Y-%m-%d-%Hh%Mm%Ss')).with_suffix(".h5")
                if bytes_received == bytes_expected:
                    with open(f"{file_path!s}", 'wb') as f:
                        f.write(data)
                        msg = "OK"
                scs.write(msg)
                msg = scs.read_wait()
                print(f"File download: {msg}")


parse_arguments()
if __name__ == "__main__":
    connect()
