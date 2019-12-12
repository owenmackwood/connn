import socket
import logging
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from connectn.utils import configure_logging, parse_arguments
from connectn.utils import ComfyStockings, ROOT_DATA_DIR, SERVER_PROCESS_DATA_DIR

parse_arguments()
configure_logging()

if not ROOT_DATA_DIR.exists():
    ROOT_DATA_DIR.mkdir()
if not SERVER_PROCESS_DATA_DIR.exists():
    SERVER_PROCESS_DATA_DIR.mkdir()


def run_server():
    from connectn.utils import LISTEN_PORT, InactiveSocket, PLAY_ALL
    from connectn.game import run_games_process
    from multiprocessing.managers import SyncManager

    logger = logging.getLogger(__name__)

    manager = SyncManager()
    manager.start(_process_init)
    sq = mp.Queue()
    rq = manager.Queue()
    shutdown = manager.Event()
    rg = mp.Process(target=_process_init, args=(run_games_process, sq, rq, shutdown, PLAY_ALL))
    rg.start()

    logger.info("Started run_games process")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ls:
        try:
            ls.settimeout(5.0)
            ls.bind(("localhost", LISTEN_PORT))
            ls.listen(5)
            logger.info("Started server listening socket")
        except Exception:
            logger.exception("Failure when binding to the listening port.")
        else:
            updated_agent_archives = []
            running = True
            while running:
                try:
                    (cs, addr) = ls.accept()
                    logger.info("Accepted connection.")
                    handle_client(cs, updated_agent_archives)
                except socket.timeout:
                    if len(updated_agent_archives):
                        logger.info(
                            f"Server sending {len(updated_agent_archives)} new agents for game-play."
                        )
                        logger.info(f"{updated_agent_archives}")
                        sq.put(updated_agent_archives)
                        updated_agent_archives = []
                except InactiveSocket:
                    logger.exception("Connection failed")
                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt: Shutting down")
                    running = False
                except Exception:
                    logger.exception("Unexpected error, will try to keep running.")

                store_results_local(rq)
        finally:
            """
            If the port is orphaned use:
            fuser -kn tcp <port>
            """
            ls.shutdown(socket.SHUT_RDWR)
            logger.info("Closed server socket")

    logger.info("Telling run_games process to shutdown.")
    shutdown.set()
    rg.join()
    logger.info("Finished server shutdown.")


def store_results_local(rq: mp.Queue):
    from queue import Empty as EmptyQueue

    logger = logging.getLogger(__name__)
    try:
        updated_results = rq.get(block=False)
    except EmptyQueue:
        pass
    else:
        if updated_results:
            logger.info(f"Storing {len(updated_results)} result files.")

            for file_name, results_data in updated_results.items():
                with open(f"{results_file_name_local(file_name)!s}", "wb") as f:
                    f.write(results_data)
            logger.info("Finished storing the result files.")


def results_file_name_local(file_name: str) -> Path:
    return (SERVER_PROCESS_DATA_DIR / file_name).with_suffix(".h5")


def handle_client(cs: socket.socket, updated_agent_archives: list):
    from time import sleep
    from connectn.users import authenticate_user

    logger = logging.getLogger(__name__)

    with ComfyStockings(cs) as scs:
        scs.handshake_wait()

        uid_pw = scs.read_wait().split(",")
        logger.info(f"Authentication attempt by: {uid_pw[0]}")
        login_valid = len(uid_pw) == 2 and authenticate_user(*uid_pw)
        msg = "OK" if login_valid else "FAIL"
        scs.write(msg)
        if login_valid:
            logger.info("Authentication successful.")
            # User is prompted to choose upload or download
            up_or_down = scs.read_wait()
            if up_or_down == "UPLOAD":
                msg = handle_upload(scs, uid_pw[0], updated_agent_archives)
            elif up_or_down == "DOWNLOAD":
                msg = handle_download(scs, uid_pw[0])
            else:
                msg = "FAIL"
            scs.write(msg)
        else:
            logger.info("Authentication failed.")

        sleep(0.5)
        while scs.active:
            logger.info("Waiting for socket to become inactive")
            sleep(1.0)


def handle_upload(
    scs: ComfyStockings, username: str, updated_agent_archives: List[Tuple[str, str]]
) -> str:
    import os
    import tempfile

    logger = logging.getLogger(__name__)

    scs.write("READY")
    bytes_expected = int(scs.read_wait())
    logger.info(f"Now starting transfer of {bytes_expected} bytes")
    data = scs.read_wait()
    bytes_received = len(data)
    logger.info(f"Received bytes {bytes_received}, expected {bytes_expected}")
    msg = "FAIL"
    if bytes_received == bytes_expected:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            exists = None
            for pending in updated_agent_archives:
                if pending[0] == username:
                    logger.warning(
                        f"Found pending update. It will be overwritten {pending}"
                    )
                    exists = pending
                    os.remove(pending[1])
            if exists is not None:
                updated_agent_archives.remove(exists)
            updated_agent_archives.append((username, f.name))
            msg = "OK"
    return msg


def handle_download(scs: ComfyStockings, username: str) -> str:
    import os
    import time
    from connectn.utils import TOURNAMENT_FILE

    logger = logging.getLogger(__name__)

    # Client is asked whether to download their agent file, or all matches (tournament)
    msg = scs.read_wait()
    if msg == "TOURNAMENT":
        results_path = results_file_name_local(TOURNAMENT_FILE)
    else:
        results_path = results_file_name_local(username)

    t = time.gmtime(os.path.getmtime(f"{results_path!s}"))
    scs.write(time.strftime("%Y-%m-%d-%Hh%Mm%Ss", t))
    response = scs.read_wait()
    logger.info(f"Response to prompt: {response}")
    # Client is shown date/time of file creation and prompted whether to download
    if response == 'GO':
        logger.info(f"User {username} chose to proceed with download.")
        file_size = os.path.getsize(f"{results_path!s}")
        scs.write(f"{file_size!s}")
        with open(f"{results_path!s}", "rb") as f:
            scs.write(f.read())
        msg = scs.read_wait()
        if msg == "OK":
            logger.info(f"User {username} successfully received results file.")
        else:
            logger.error(f"User {username} failed to receive results file: {msg}")
    else:
        logger.info(f"User {username} chose to skip download.")
        msg = "OK"
    return msg


def _process_init(func=None, *args):
    """
    This function is used to initialize processes that need to ignore
    the KeyboardInterrupt / SIGINT signal. This includes the
    multiprocessing SyncManager and the run_games process.

    Parameters
    ----------
    func : Callable, optional
    args : Positional arguments for func
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if func is not None:
        func(*args)


if __name__ == "__main__":
    run_server()
