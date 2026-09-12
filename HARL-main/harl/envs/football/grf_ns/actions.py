"""GRF's default action set, the compass group, and gate 4.

The eight movement actions of Google Research Football are sticky compass
headings 45 degrees apart.  Listed in the engine's own order they form a cycle,
so "rotate the heading by k steps" is ``(idx + k) mod 8`` -- a group action with
an exact inverse and no saturation.  That group is what makes this instance the
INVERTIBLE cell of the classification (II.6, first row).

P-3.4 / II.9 gate 4: *"the basis's ordering of options must match the
environment's own -- a permuted order points every channel at the wrong element
while looking perfectly healthy."*  Here the analogue is the direction table
below.  ``verify_action_table()`` checks it against the installed gfootball at
construction and aborts on mismatch; the offline suite checks the geometry of
the table (consecutive entries 45 degrees apart, unit length).

Coordinates follow the GRF observation frame: x runs from the left goal (-1) to
the right goal (+1); y runs from the TOP touchline (-0.42) to the BOTTOM one
(+0.42).  So ``action_top`` is -y and ``action_bottom`` is +y.  ``smoke.py``
confirms the handedness empirically by driving a lone player.

numpy only.
"""

import numpy as np

__all__ = [
    "IDLE", "DIR_FIRST", "DIR_LAST", "LONG_PASS", "HIGH_PASS", "SHORT_PASS",
    "SHOT", "SPRINT", "RELEASE_DIRECTION", "RELEASE_SPRINT", "SLIDING",
    "DRIBBLE", "RELEASE_DRIBBLE", "N_ACTIONS", "N_DIRS", "DIR_NAMES", "DIR_VEC",
    "ACTION_NAMES", "is_direction", "heading_of", "rotate", "verify_action_table",
]

# ---------------------------------------------------------------------------
# gfootball.env.football_action_set.action_set_dict["default"], by index
# ---------------------------------------------------------------------------
IDLE = 0
DIR_FIRST, DIR_LAST = 1, 8            # left .. bottom_left, inclusive
LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT = 9, 10, 11, 12
SPRINT = 13
RELEASE_DIRECTION = 14
RELEASE_SPRINT = 15
SLIDING = 16
DRIBBLE = 17
RELEASE_DRIBBLE = 18
N_ACTIONS = 19
N_DIRS = 8

ACTION_NAMES = (
    "idle", "left", "top_left", "top", "top_right", "right", "bottom_right",
    "bottom", "bottom_left", "long_pass", "high_pass", "short_pass", "shot",
    "sprint", "release_direction", "release_sprint", "sliding", "dribble",
    "release_dribble",
)
DIR_NAMES = ACTION_NAMES[DIR_FIRST:DIR_LAST + 1]

_S = 1.0 / np.sqrt(2.0)
#: Unit heading vectors in the GRF frame, in the engine's index order.  Each
#: entry is +45 degrees (in atan2(y, x)) from the previous one, so the table IS
#: the cyclic group Z_8 and ``rotate`` below is its action.
DIR_VEC = np.array([
    [-1.0, 0.0],      # left
    [-_S, -_S],       # top_left
    [0.0, -1.0],      # top
    [_S, -_S],        # top_right
    [1.0, 0.0],       # right
    [_S, _S],         # bottom_right
    [0.0, 1.0],       # bottom
    [-_S, _S],        # bottom_left
], dtype=np.float64)


def is_direction(a):
    a = np.asarray(a)
    return (a >= DIR_FIRST) & (a <= DIR_LAST)


def heading_of(action):
    """Direction index 0..7 for a movement action, else -1."""
    a = int(action)
    return a - DIR_FIRST if DIR_FIRST <= a <= DIR_LAST else -1


def rotate(head_idx, k):
    """Rotate a heading index by ``k`` compass steps.  Exact group action:
    ``rotate(rotate(h, k), -k) == h`` for every h, k."""
    return int((int(head_idx) + int(k)) % N_DIRS)


def verify_action_table():
    """Gate 4 -- abort if the installed gfootball orders its actions differently.

    Returns a one-line description.  If gfootball is not importable (the offline
    suite) the geometric half of the check still runs and the result says so.
    """
    # geometry of the table itself: unit vectors, consecutive entries 45 deg apart
    norms = np.linalg.norm(DIR_VEC, axis=1)
    assert np.allclose(norms, 1.0), "direction table is not unit length"
    ang = np.degrees(np.arctan2(DIR_VEC[:, 1], DIR_VEC[:, 0]))
    step = (np.roll(ang, -1) - ang) % 360.0
    assert np.allclose(step, 45.0), "direction table is not a 45-degree cycle: %s" % step
    try:
        from gfootball.env import football_action_set as fas
    except Exception:  # pragma: no cover -- offline
        return "action table: geometry OK; gfootball not importable, engine order unchecked"
    names = [str(getattr(a, "_name", a)) for a in fas.action_set_dict["default"]]
    if len(names) != N_ACTIONS or tuple(names) != ACTION_NAMES:
        raise AssertionError(
            "GATE 4 FAILED: gfootball's default action set is %s, this module "
            "assumes %s.  Every rotation would point at the wrong action while "
            "every diagnostic looked healthy." % (names, list(ACTION_NAMES)))
    return "action table: matches gfootball default action set (%d actions, 8 headings)" % N_ACTIONS
