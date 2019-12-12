import argparse
from pathlib import Path
from connectn.utils import ComfyStockings

group_name = "group_a"
password = "rgIWtVRp"
module_location = Path("~/codebase/connn/connectn/agents/agent_mcts")
results_location = Path.home() / "tournament"

if not results_location.exists():
    results_location.mkdir()


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
    import socket
    from connectn.utils import LISTEN_PORT

    upload = False
    inp = input("Upload agent or download results? u / [d] ").lower()
    while inp not in ("", "u", "d"):
        inp = input().lower()
    if inp == "u":
        upload = True

    download_agent = False
    if not upload:
        inp = input("Download tournament or agent file? t / [a] ").lower()
        while inp not in ("", "t", "a"):
            inp = input().lower()
        download_agent = inp != "t"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cs:
        cs.connect(("localhost", LISTEN_PORT))
        with ComfyStockings(cs) as scs:
            scs.handshake_wait()

            print("Sending authentication request.")
            scs.write(f"{group_name},{password}")

            msg = scs.read_wait()
            if "OK" != msg:
                raise Exception(f"Authentication: {msg}")

            scs.write("UPLOAD" if upload else "DOWNLOAD")
            if upload:
                handle_upload(scs)
            else:
                handle_download(scs, download_agent)


def handle_upload(scs: ComfyStockings):
    import os
    import tempfile
    import tarfile

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


def handle_download(scs: ComfyStockings, download_agent: bool):
    from connectn.utils import TOURNAMENT_FILE

    scs.write("AGENT" if download_agent else "TOURNAMENT")

    file_time = scs.read_wait()
    file_name =  "-".join((group_name if download_agent else TOURNAMENT_FILE, file_time))
    file_path = (results_location / file_name).with_suffix(".h5")

    if file_path.exists():
        scs.write("STOP")
        print("You already have this version of the file. "
              "If you want to download it again, rename/delete the existing one.")
        msg = scs.read_wait()
        print(f"Skipped download: {msg}")
    else:
        scs.write("GO")
        bytes_expected = int(scs.read_wait())
        print(f"Now starting transfer of {bytes_expected} bytes")
        data = scs.read_wait()
        bytes_received = len(data)
        print(f"Received bytes {bytes_received}, expected {bytes_expected}")
        msg = "FAIL"
        if bytes_received == bytes_expected:
            with open(f"{file_path!s}", "wb") as f:
                f.write(data)
                msg = "OK"
        scs.write(msg)
        msg = scs.read_wait()
        print(f"File download: {msg}")


parse_arguments()
if __name__ == "__main__":
    connect()
