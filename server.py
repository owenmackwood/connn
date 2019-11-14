import socket


def run_server():
    import logging
    import multiprocessing as mp
    from connectn.utils import LISTEN_PORT, InactiveSocket
    from connectn.game import run_games

    q = mp.Queue()
    rg = mp.Process(target=run_games, args=(q,))
    rg.start()
    print('Started run_games process')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ls:
        try:
            ls.settimeout(5.)
            ls.bind(('localhost', LISTEN_PORT))
            ls.listen(5)
            print('Started server listening socket')

            updated_agent_archives = []
            while True:
                try:
                    (cs, addr) = ls.accept()
                    print('Accepted connection.')
                    handle_client(cs, updated_agent_archives)
                except InactiveSocket as e:
                    print(f'Connection failed: {e}')
                except socket.timeout:
                    if len(updated_agent_archives):
                        print(f'Server sending {len(updated_agent_archives)} new agents for game-play.')
                        q.put(updated_agent_archives)
                        updated_agent_archives = []
                except Exception as e:
                    print(f'Unexpected error {type(e)}')
                    print('Will try to keep running.')
        except:
            logging.exception('Failure in the server')
        finally:
            ls.shutdown(socket.SHUT_RDWR)
            print("Closed server socket")
            '''
            If the port is orphaned use:
            fuser -kn tcp <port>
            '''
            q.put('SHUTDOWN')
            rg.join()


def handle_client(cs: socket.socket, updated_agent_archives: list):
    import os
    from time import sleep
    import tempfile
    from connectn.users import authenticate_user
    from connectn.utils import ComfyStockings

    with ComfyStockings(cs) as scs:
        scs.handshake_wait()

        uid_pw = scs.read_wait().split(',')
        print(f"Authentication attempt by: {uid_pw[0]} {len(uid_pw)}")
        login_valid = len(uid_pw) == 2 and authenticate_user(*uid_pw)
        msg = 'OK' if login_valid else 'FAIL'
        scs.write(msg)
        if login_valid:
            bytes_expected = int(scs.read_wait())
            print(f'Now starting transfer of {bytes_expected} bytes')
            data = scs.read_wait()
            bytes_received = len(data)
            print(f'Received bytes {bytes_received}, expected {bytes_expected}')
            msg = 'FAIL'
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
                    msg = 'OK'

            scs.write(msg)
        sleep(.5)
        while scs.active:
            print('Waiting for socket to become inactive')
            sleep(1.)


if __name__ == '__main__':
    run_server()

