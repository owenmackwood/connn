import logging
import importlib
import hashlib
import pickle
import os
from typing import Tuple, Dict, List
from types import ModuleType
from connectn.utils import KEY_SALT_FILE, AGENTS_PER_USER

user_auth: Dict[str, Tuple[bytes, bytes]] = {}


def load_user_auth() -> Dict[str, Tuple[bytes, bytes]]:
    global user_auth
    if len(user_auth) < 1 and KEY_SALT_FILE.exists():
        with open(f"{KEY_SALT_FILE!s}", "rb") as f:
            user_auth.update(pickle.load(f))
        if AGENTS_PER_USER > 1:
            users_auth = [(user, auth) for user, auth in user_auth.items()]
            for i in range(2, AGENTS_PER_USER+1):
                for user, auth in users_auth:
                    user_auth[f"{user}{i}"] = auth
    return user_auth


def agents() -> List[str]:
    builtin_agents = [
        "agent_random",
        "agent_columns",
        "agent_rows",
        # "agent_mcts",
        # "agent_mmab",
        "agent_fail",
    ]
    agent_names = builtin_agents + list(load_user_auth().keys())
    return agent_names


def import_agents(agent_modules: Dict[str, ModuleType]) -> Dict[str, ModuleType]:
    logger = logging.getLogger(__name__)

    new_modules = dict()
    for agent in agents():
        name = ".".join(("connectn", "agents", agent))
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
        except Exception:
            logger.exception(f'Failed to import module {name}')
    return new_modules


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)


def authenticate_user(user: str, password: str) -> bool:
    import string

    logger = logging.getLogger(__name__)
    all_user_auth = load_user_auth()
    user_version = user[-1]
    if user_version in string.digits:
        if 0 < int(user_version) <= AGENTS_PER_USER:
            user = user[:-1]
        else:
            logger.warning(f"User {user} provided an illegal version.")
    if user in all_user_auth:
        key, salt = all_user_auth[user]
        return key == hash_password(password, salt)
    return False


def generate_users(num_users: int, pw_length: int = 8, append: bool = True):
    import random
    import string
    from pathlib import Path

    user_pw_plain = Path.home() / "users_passwords.txt"

    n_users = 0
    new_users_key_salt = {}
    if append:
        new_users_key_salt = load_user_auth()
        n_users = len(new_users_key_salt)
    new_users = [
        f'group_{chr(ord("a")+g)}' for g in range(n_users, n_users + num_users)
    ]

    new_users_pw = {}

    for user in new_users:
        pw = "".join(
            random.SystemRandom().choice(string.ascii_letters + string.digits)
            for _ in range(pw_length)
        )
        salt = os.urandom(32)
        key = hash_password(pw, salt)
        new_users_pw[user] = pw
        new_users_key_salt[user] = key, salt

    with open(f"{user_pw_plain!s}", "a" if append else "w") as f:
        for user, pw in new_users_pw.items():
            f.write(f"{user} {pw}\n")

    with open(f"{KEY_SALT_FILE!s}", "wb") as f:
        pickle.dump(new_users_key_salt, f)


if __name__ == "__main__":
    generate_users(1, append=True)
