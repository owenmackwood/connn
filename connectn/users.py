import importlib
import hashlib
import pickle
import os
from typing import Tuple, Dict
from types import ModuleType
from connectn.utils import DATA_DIR

key_salt_file = os.path.join(DATA_DIR, 'keys_salts')
user_auth = {}


def load_user_auth() -> Dict[str, Tuple[bytes, bytes]]:
    global user_auth
    if len(user_auth) < 1 and os.path.exists(key_salt_file):
        with open(key_salt_file, 'rb') as f:
            user_auth.update(pickle.load(f))
    return user_auth


def agents():
    # agents = ['agent_mctsh', 'agent_mcts'] + list(load_user_auth().keys())
    agent_names = ['agent_random', 'agent_columns', 'agent_rows', 'agent_mcts'] + list(load_user_auth().keys())
    return agent_names


def import_agents(agent_modules: Dict[str, ModuleType]) -> Dict[str, ModuleType]:
    new_modules = dict()
    for agent in agents():
        name = '.'.join(('connectn', 'agents', agent))
        new_load = agent not in agent_modules
        try:
            if new_load:
                module_obj = importlib.import_module(name)
            else:
                module_obj = importlib.reload(agent_modules[agent])
            new_modules[agent] = module_obj
        except ModuleNotFoundError:
            pass
            # print(f'No module provided yet by {name}')
        except Exception as e:
            print(f'Failed to import module {name}, with error: {e}')
    return new_modules


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)


def authenticate_user(user: str, password: str) -> bool:
    user_auth = load_user_auth()
    if user in user_auth:
        key, salt = user_auth[user]
        return key == hash_password(password, salt)
    return False


def generate_users(num_users: int, pw_length: int=8, append: bool=True):
    import random
    import string
    user_pw_plain = '~/ppp_user_passwords.txt'

    n_users = 0
    new_users_key_salt = {}
    if append:
        new_users_key_salt = load_user_auth()
        n_users = len(new_users_key_salt)
    new_users = [f'group_{chr(ord("a")+g)}' for g in range(n_users, n_users+num_users)]

    new_users_pw = {}

    for user in new_users:
        pw = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(pw_length))
        salt = os.urandom(32)
        key = hash_password(pw, salt)
        new_users_pw[user] = pw
        new_users_key_salt[user] = key, salt

    with open(os.path.expanduser(user_pw_plain), 'a' if append else 'w') as f:
        for user, pw in new_users_pw.items():
            f.write(f'{user} {pw}\n')

    with open(key_salt_file, 'wb') as f:
        pickle.dump(new_users_key_salt, f)


if __name__ == '__main__':
    generate_users(1, append=False)
