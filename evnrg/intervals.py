import numba as nb
import numpy as np

from .eligibility import EligibilityRules, Eligibility
from .bracket import Bracket, make_bracket


@nb.njit
def next_charging_opportunity(current_bracket: Bracket,
                              distance_array: np.array,
                              rules: EligibilityRules):

    if not (distance_array.ndim == 1):
        raise ValueError('Array of zero length supplied.')
    i = current_bracket.end
    out = None
    if current_bracket.end < distance_array.shape[0]:
        while i < distance_array.shape[0]:
            # if we're stopped:
            if distance_array[i] == 0.:
                bracket = make_bracket(distance_array, i, rules)

                if bracket.eligibility in {Eligibility.HOME, Eligibility.AWAY}:
                    out = bracket
                    break
            i += 1

    if bracket is None:
        out = Bracket(begin=distance_array.shape[0],
                      end=distance_array.shape[0])
    return out


@nb.njit
def distance_between_brackets(distance_array: np.array, b_start: Bracket,
                              b_end: Bracket):
    return distance_array[b_start.end:b_end.begin].sum()


@nb.njit
def is_last_bracket(distance_array: np.array, bkt: Bracket):
    return bkt.end >= distance_array.shape[0]


@nb.njit
def rewrite_deferred_trips(distance_array: np.array,
                           deferred_array: np.array,
                           ev_range: float, start_bracket: Bracket,
                           rules: EligibilityRules):

    # Only worry about doing this if:
    # 1: The vehicle is stopped
    # and 2: The ehicle is a BEV. PHEVs and ICEVs can leave any time.
    # and 3: The vehicle is about to leave, i.e.: the next interval
    # is a distance > 0

    current_bracket = start_bracket

    target_bracket = start_bracket

    while not is_last_bracket(distance_array, target_bracket):

        target_bracket = next_charging_opportunity(
            current_bracket,
            distance_array,
            rules
            )

        # Check to make sure the bracket a good one
        if target_bracket.begin < distance_array.shape[0]:
            dist = distance_between_brackets(
                distance_array,
                current_bracket,
                target_bracket)
            # If the BEV won't make it, rewrite!
            if dist > ev_range:
                x = current_bracket.end
                y = target_bracket.begin
                deferred_array[x:y] = distance_array[x:y]
                distance_array[x:y] = 0

                current_bracket = make_bracket(
                    distance_array,
                    current_bracket.index,
                    rules
                )
            # if the BEV can make it, breaks the loop!
            else:
                break

    return current_bracket
