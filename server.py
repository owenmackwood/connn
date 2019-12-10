import socket
import logging
from connectn.utils import configure_logging, parse_arguments

parse_arguments()
configure_logging()


def run_server():
    import multiprocessing as mp
    from connectn.utils import LISTEN_PORT, InactiveSocket
    from connectn.game import run_games

    logger = logging.getLogger(__name__)
    q = mp.Queue()
    rg = mp.Process(target=run_games, args=(q,))
    rg.start()
    print("Started run_games process")
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
            while True:
                try:
                    (cs, addr) = ls.accept()
                    logger.info("Accepted connection.")
                    handle_client(cs, updated_agent_archives)
                except InactiveSocket:
                    logger.exception("Connection failed")
                except socket.timeout:
                    if len(updated_agent_archives):
                        logger.info(
                            f"Server sending {len(updated_agent_archives)} new agents for game-play."
                        )
                        q.put(updated_agent_archives)
                        updated_agent_archives = []
                except Exception:
                    logger.exception("Unexpected error, will try to keep running.")
        finally:
            ls.shutdown(socket.SHUT_RDWR)
            logger.info("Closed server socket")
            """
            If the port is orphaned use:
            fuser -kn tcp <port>
            """
            q.put("SHUTDOWN")
            rg.join()
            print("Finished shutdown")


def handle_client(cs: socket.socket, updated_agent_archives: list):
    import os
    from time import sleep
    import tempfile
    from connectn.users import authenticate_user
    from connectn.utils import ComfyStockings

    logger = logging.getLogger(__name__)
    with ComfyStockings(cs) as scs:
        scs.handshake_wait()

        uid_pw = scs.read_wait().split(",")
        logger.info(f"Authentication attempt by: {uid_pw[0]} {len(uid_pw)}")
        login_valid = len(uid_pw) == 2 and authenticate_user(*uid_pw)
        msg = "OK" if login_valid else "FAIL"
        scs.write(msg)
        if login_valid:
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
                        if pending[0] == uid_pw[0]:
                            exists = pending
                            os.remove(pending[1])
                    if exists is not None:
                        updated_agent_archives.remove(exists)
                    updated_agent_archives.append((uid_pw[0], f.name))
                    msg = "OK"

            scs.write(msg)
        sleep(0.5)
        while scs.active:
            logger.info("Waiting for socket to become inactive")
            sleep(1.0)


if __name__ == "__main__":
    run_server()
