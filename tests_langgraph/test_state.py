from app.state import REQUIRED_KEYS, make_initial_state


def test_make_initial_state_has_required_keys():
    state = make_initial_state(user_id="u1", session_id="s1", user_input="hello")
    for key in REQUIRED_KEYS:
        assert key in state
