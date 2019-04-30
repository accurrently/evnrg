from typing import NamedTuple

import numpy as np
import numba as nb

from .eligibility import EligibilityRules
from .eligibility import check_mask
from .status import Status


@nb.jitclass([
    ('index', nb.int64),
    ('begin', nb.int64),
    ('status', nb.uint8),
    ('index_distance', nb.float32),
    ('total_distance', nb.float32)
])
class Bracket(object):
    """A jitclass that provides information about the current drive or stop.

    Attributes:
        index (numba.int64): The index in the distance array that the bracket
            was created.
        begin (numba.int64): The beginning of the bracket
        end (numba.int64): The end of the bracket. Also indicates the following
            bracket's `begin`.
        status (numba.uint8): The current 'Eligibility` for the bracket.
        index_distance (numba.float32): The current distance at the index (km).
        total_distance (numba.float32): The total distance traveled in the
            bracket (km).

    Args:
        idx (int): The current index
        vidx (int): The fleet index (column index) of the vehicle.
        distance_array (numpy.array): A 1-D distance interval array.
        rules (EligibilityRules): The EligibilityRules object that
            determines charging eligibility.
    """

    def __init__(self, idx, vidx, distance_array, rules):
        """Creates a Bracket object"""

        self.index = idx
        self.end = idx
        self.begin = idx
        self.total_distance = 0.
        self.index_distance = 0.
        self.status = Status.NONE

        max_len = distance_array.shape[0]

        driving = False
        if distance_array[idx] > 0.:
            driving = True

        # Only do the following if idx is inside the array.
        # If it is equal or greater than the array length,
        # we have a Bracket that represents the end of the array
        if idx < max_len:

            # Find the beginning
            while self.begin > 0:
                if driving:
                    if not (distance_array[self.begin - 1] > 0.):
                        break
                else:
                    if distance_array[self.begin - 1] > 0.:
                        break
                self.begin -= 1

            # Find the end
            while self.end < max_len:
                if driving:
                    if not (distance_array[self.end] > 0.):
                        break
                else:
                    if distance_array[self.end] > 0.:
                        break
                self.end += 1

            self.index_distance = distance_array[idx]

            length = self.end - self.begin
            # See if we've been stopped long enough to trigger
            # a charge opportunity
            home = False
            if rules.home_threshold > 0 and length > rules.home_threshold:
                home = True

            away = False
            if rules.away_threshold > 0 and length > rules.away_threshold:
                away = True

            if home and away:
                self.status = rules.threshold_priority
            elif home:
                self.status = Status.HOME_ELIGIBLE
            elif away:
                self.status = Status.AWAY_ELIGIBLE

            # Masks override any time-based rules
            # This allows for time-fencing and geo-fencing at the data
            # preparation level.
            charge_mask = check_mask(self.begin, self.end, vidx, rules)

            if charge_mask in {Status.HOME_ELIGIBLE, Status.AWAY_ELIGIBLE}:
                self.status = charge_mask

            self.total_distance = distance_array[self.begin:self.end].sum()
        # Endif

    @property
    def length(self):
        """Returns the number of intervals in the bracket"""
        return self.end - self.begin

    @property
    def driving(self):
        """Returns if the bracked is a driving one."""
        return self.index_distance > 0.

    @property
    def stopped(self):
        """Returns if the bracked is a stop."""
        return not self.is_driving


# Deprecated the NamedTuple implementation
class BracketNT(NamedTuple):
    """Deprecated, but kept for reference"""
    begin: int = -1
    end: int = -1
    index: int = -1
    length: int = 0
    mask: bool = False
    status: Status = Status.STOPPED
    dist: float = 0.
    idist: float = 0.
    stopped: bool = True


def update_bracket(bkt: BracketNT, d: dict):
    """Deprecated"""
    x = bkt._todict
    _d = {k: d[k] for k, v in Bracket.items() if k in Bracket._fields}
    x.update(_d)

    return Bracket(**x)


@nb.njit
def make_bracket(arr: np.array, index: int, einfo: EligibilityRules):
    """Deprecated"""

    alen = arr.shape[0]
    mask = einfo.home_mask

    assert alen == mask.shape[0], \
        'Mask and distance array lengths do not match.'

    e = Status.NONE
    idist = arr[index]
    stopped = True
    # If we're not currently stopped, don't bother.
    if idist > 0:
        e = Status.DRIVING
        stopped = False

    begin = index + 0
    end = index + 0

    def _test_(a, b):
        x = a > 0.  # Test for driving
        if b:  # If stopped
            x = not (a > 0.)  # Then invert test
        return x

    # search backwards
    while begin > 0:
        if _test_(arr[begin - 1], stopped):
            break
        begin -= 1

    # Search forwards
    while end < alen:
        if _test_(arr[end], stopped):
            break
        end += 1

    home_mask_match = (mask[begin:end].sum() > 0)
    dist = arr[begin:end].sum()
    length = end - begin

    if home_mask_match:
        e = Status.HOME_ELIGIBLE

    elif stopped:
        for p in einfo.priority:

            if p == Status.HOME_ELIGIBLE:
                if length >= einfo.home_threshold:
                    e = Status.HOME_ELIGIBLE

            elif p == Status.AWAY_ELIGIBLE:
                if length >= einfo.away_threshold:
                    e = Status.AWAY_ELIGIBLE

    return BracketNT(
        begin=begin,
        end=end,
        index=index,
        length=length,
        mask=home_mask_match,
        status=e,
        dist=dist,
        idist=idist,
        stopped=stopped
    )
    