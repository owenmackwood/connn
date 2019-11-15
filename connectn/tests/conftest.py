import pytest

@pytest.fixture
def empty_board():
    from connectn.game import initialize_game_state
    return initialize_game_state()


@pytest.fixture
def full_board_p1():
    from connectn.game import initialize_game_state, PLAYER1
    board = initialize_game_state()
    board.fill(PLAYER1)
    return board


@pytest.fixture
def full_board_p2():
    from connectn.game import initialize_game_state, PLAYER2
    board = initialize_game_state()
    board.fill(PLAYER2)
    return board

