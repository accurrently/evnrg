from typing import NamedTuple

import numpy as np
import numba as nb

from .status import Status

__all__ = [
    'EligibilityRules',
    'ECode'
]

class EligibilityRules(NamedTuple):

    mask: np.array = np.empty((0, 0), dtype=np.bool)
    threshold_priority: int = Status.HOME_ELIGIBLE
    away_threshold: int = 0
    home_threshold: int = 0


@nb.njit
def check_mask(begin: int, end: int, col: int, rules: EligibilityRules):

    out = Status.INELIGIBLE
    if end < rules.mask.shape[0] and begin >= 0:
        if rules.mask.ndim == 2:
            out = max(rules.mask[begin:end, col].max(), 0)
        elif rules.mask.ndim == 1:
            out = max(rules.mask[begin:end].max(), 0)

    return out


class ECode(NamedTuple):
    """Small `NamedTuple` to indicate the stop's eligibility.
    
    Attributes:
        begin_index (int): Beginning index of the stop.
        end_index (int): Ending inde xof the stop.
            This is also the index of the beginning of the next
            trip.
        code (Status): The status of the a stop. Will be one of 
            `Status.INELIGIBLE`, `Status.HOME_ELIGIBLE`, or 
            `Status.AWAY_ELIGIBLE`. If the stop is not actually a stop,
            `Status.DRIVING` is used.

    """
    begin_index: int
    end_index: int
    code: int


@nb.jit
def stop_eligibility(distance_array: np.array,
                     begin: int,
                     vidx: int,
                     rules: EligibilityRules,
                     use_mask: bool = True):
    """Returns an `ECode` code indicating the charge eligibility."""

    status = Status.DRIVING
    end = begin
    if distance_array[begin] == 0:

        status = Status.INELIGIBLE

        alen = distance_array.shape[0]

        while end < alen and not (distance_array[end] > 0):
            end += 1

        length = end - begin
        # See if we've been stopped long enough to trigger
        # a charge opportunity
        home = False
        if rules.home_threshold > 0 and length > rules.home_threshold:
            home = True

        away = False
        if rules.away_threshold > 0 and length > rules.away_threshold:
            away = True

        if home and away:
            status = rules.threshold_priority
        elif home:
            status = Status.HOME_ELIGIBLE
        elif away:
            status = Status.AWAY_ELIGIBLE

        # Masks override any time-based rules
        # This allows for time-fencing and geo-fencing at the data
        # preparation level.
        if use_mask:
            charge_mask = check_mask(begin, end, vidx, rules)

            allowed_mask_values = (
                Status.HOME_ELIGIBLE,
                Status.AWAY_ELIGIBLE
            )

            if charge_mask in allowed_mask_values:
                status = charge_mask

    return ECode(
        begin_index=begin,
        end_index=end,
        code=status
    )
