import numpy as np
import numba as nb
import pandas as pd
from typing import List
import random

from evnrg.common.status import Status
from .eligibility import (
    EligibilityRules,
    ECode
)
from .eligibility import stop_eligibility
from evnrg.common.powertrain import Powertrain, PType
from .bracket import Bracket
from evnrg.common.plug import DCPlug

__all__ = [
    'Vehicle'
]

@nb.njit
def next_stop(distance_a: np.array, begin: int):
    i = begin
    length = distance_a.shape[0]
    d = distance_a[begin]
    if d > 0:
        while i < length and d > 0:
            d = distance_a[i]
            i += 1
    return ECode(
        begin_index=begin,
        end_index=i,
        code=Status.DRIVING
    )


@nb.njit
def next_trip(distance_a: np.array, begin: int):
    i = begin
    length = distance_a.shape[0]
    while i < length and distance_a[i] == 0.:
        i += 1
    return i


@nb.njit
def next_charging_opportunity(distance_array: np.array,
                              begin: int,
                              vidx: int, rules: EligibilityRules):
    """Finds the next Bracket that's a viable charging opportunity.

    This is a JITted function.

    Args:
        current_bracket (Bracket): The `Bracket` to start searching from.
        distance_array (numpy.array): A 1-D Numpy distance array.
        vidx (int): The vehicle column index to use for mask lookup
        rules (EligibilityRules): The `EligibilityRules` object to use
    """

    if not (distance_array.ndim == 1):
        raise ValueError('Array of zero length supplied.')

    i = begin
    alen = distance_array.shape[0]
    code = ECode(alen, alen, Status.HOME_ELIGIBLE)
    if distance_array[begin] == 0. and distance_array[begin + 1] > 0.:
        while i < alen:
            if distance_array[i] > 0.:
                i += 1
            else:
                code = stop_eligibility(distance_array, i, vidx, rules)

                acceptable_codes = (
                    Status.HOME_ELIGIBLE,
                    Status.AWAY_ELIGIBLE
                )

                if code.code in acceptable_codes:
                    break

                else:
                    i = code.end_index

    return code


@nb.njit
def distance_between_brackets(distance_array: np.array, b_start: Bracket,
                              b_end: Bracket):
    """Returns the distance between two brackets

    Args:
        distance_array (numpy.array): A 1-D distance array
        b_start (Bracket): The `Bracket` to start from
        b_end (Bracket): The `Bracket` to end at

    Returns:
        A `float` of the total distance traveled in the bracket.
    """
    return distance_array[b_start.end:b_end.begin].sum()


@nb.njit
def is_last_bracket(distance_array: np.array, bkt: Bracket):
    """Returns if the bracket is the last bracket in thedistance array

    Args:
        distance_array (numpy.array): The 1-D `numpy.array` to test against
        bkt (Bracket): The `Bracket` to test

    Returns:
        `bool`
    """
    return bkt.end >= distance_array.shape[0]


@nb.njit
def rewrite_deferred_trips(distance_array: np.array,
                           deferred_array: np.array,
                           begin: int,
                           target: int,
                           begin_ev_range: float,
                           max_ev_range: float,
                           range_added_per_interval: float = 0.,
                           soc_buffer: float = 0.):
    """Searches for next eligible stop for BEVs, and defers trips as necessary.

    If a BEV's range can't achieve the distance, the trips are deferred until
    the distance is less than the range available. Additionally, This allows
    vehicles in a queue to stay in it, or vehicles to continue charging.

    Note:
        This function modifies the distance and deferred arrays in-place.

    Args:
        distance_array (numpy.array): The distance array to use
        deferred_array (numpy.array): The array that will be used to record
            deferred distance
        begin (int): The index to begin the operation at
        vidx (int): Vehicle column index (fleet_id)
        begin_status (int): The `Status` at the beginning index.
        begin_ev_range (float): Current range remaining in vehicle's battery
        max_ev_range (float): Maximum range the vehicle's battery can have
        rules (EligibilityRules): the `EligibilityRules` to use
        range_added_per_interval (:obj: `float`, optional): Allows range to be
            added if the current state is `Status.CONNECTED`.

    Returns:
        A revised `Status`.

    """

    # Only worry about doing this if:
    # 1: The vehicle is stopped
    # and 2: The ehicle is a BEV. PHEVs and ICEVs can leave any time.
    # and 3: The vehicle is about to leave, i.e.: the next interval
    # is a distance > 0

    max_len = distance_array.shape[0]

    if begin < max_len - 1:

        ev_range = begin_ev_range

        socmult = 1.0 - soc_buffer

        required_dist = distance_array[begin:target].sum()

        i = begin

        while i < target and (ev_range * socmult) < required_dist:
            # is the index advanced to a trip?
            if distance_array[i] > 0:
                j = i
                # Find the end of the trip
                while distance_array[j] > 0:
                    deferred_array[j] = distance_array[j]
                    distance_array[j] = 0.0
                    j += 1
                # Defer the trip in this window
                required_dist = distance_array[begin:target].sum()

                # Add range if we're connected
                #range_added = 0.
                #if range_added_per_interval > 0.:
                #    range_added = (j - i) * range_added_per_interval
                #    ev_range = min(range_added + ev_range, max_ev_range)
                # Move up the index
                i = j
            else:
                # Iterate through a non-eligible stop
                i += 1
        
    return distance_array, deferred_array


@nb.njit
def drive_to_next_stop(begin: int, distance_a: np.array, fuel_a: np.array,
                       battery_a: np.array, battery_start: float,
                       ev_efficiency: float, fuel_efficiency: float):
    """Efficiently drives a vehicle through a drive bracket and stores energy data.

    Args:
        distance_a (numpy.array): 1-D Numpy distance array
        fuel_a (numpy.array); 1-D Numpy array to record fuel use
        battery_a (numpy.array): 1-D Numpy array that holds battery state
        battery_start (float): The starting battery energy for the vehicle
        ev_efficiency (float): The efficiency of electric operation in km/kWh.
            For ICEVs, this should be zero.
        fuel_efficiency (float): The efficiency of combustion operation
            in km/L. For BEVs, this should be zero.

    Returns:
        An `int` indicating the number of intevals traveled.

    Raises:
        ValueError: if the EV distance and Ice distance do not add up to the
            interval's distance (meaning the EV could not make it).

    """
    error_ = .001
    # Double check we're driving
    i = begin
    if distance_a[i] > 0.:
        max_len = distance_a.shape[0]
        while (i < max_len) and (distance_a[i] > 0.):

            distance = distance_a[i]
            e_dist = 0.
            fuel_dist = 0.

            # Check if we have a pev
            if ev_efficiency > 0.:
                # Do this to make sure we don't try to access index of -1
                battery_state = battery_start
                if i > 0:
                    battery_state = battery_a[i - 1]

                # Only continue if we actually have battery power left
                if battery_state > 0.:
                    # Process battery
                    battery_required = distance / ev_efficiency
                    battery_used = min(battery_state, battery_required)
                    battery_a[i] = battery_state - battery_used
                    e_dist = round(battery_used * ev_efficiency, 4)

            if fuel_efficiency > 0.:

                fuel_dist = distance - e_dist

                fuel_a[i] = fuel_dist / fuel_efficiency

            i += 1

    return i


class Vehicle(object):
    """The primary driving logic object.

    The `Vehicle` object keeps track of all data and operations specific
    to an individual vehicle. Many of the operations are sped up through
    external JITed functions.

    Attributes:
        fleet_id (int): The column index of the vehicle for use when
            referencing 2-D array or positions in a `Fleet` object.
        powertrain (Powertrain): The `'Powertrain` used for this vehicle's
            energy calculations.
        max-soc (float): The maximum state of charge (0.0 to 1.0) this
            vehicle's battery may reach. This will be handy for modeling
            battery degradation. For now, it defaults to 1.
        evse_power (float): The effective power level in kW that this vehicle
            is connected to. If the vehicle is not connected to EVSE, this
            value is 0.
        distance_a (numpy.array): A 1-D array of type `numpy.float32` that
            contains the interval distance data. Unless deferring and
            rewriting trips, this array should generally be treated as
            read-only.
        fuel_burned_a (numpy.array): A 1-D array of type `numpy.float32` that
            holds fuel consumption data.
        deferred_a (numpy.array): A 1-D array of type `numpy.float32` that
            holds deferred distance. When BEVs have to defer trips, the
            distances are copied to this array before the window in
            `distance_a` is set to zero. This offers useful data about trip
            completion by BEVs.
        battery_a (numpy.array): A 1-D array of type 'numpy.float32` that
            reflects the battery energy of the vehicle over time.
        idx (int): The internal index for the `Vehicle` object.
        begin_energy (float): The energy the vehicle starts with. This value
            is referenced in battery operations when `idx` is 0.
        status (int): indicates the current status of the vehicle. The value
            is held by an `int`, but is generally set by referencing the
            `Status` object (which is an `enum.IntEnum`). Do not try to
            evaluate this value on its own. Instead use `Status`
            (e.g. ``vehicle.status == Status.CONNECTED``

    Args:
        fleet_index (int): The vehicle column index from a `Fleet` object.
        ptrain (Powertrain): The `Powertrain` object that will be used with
            this vehicle.
        distance (numpy.array): A 1-D `numpy.array` of type `numpy.float32`
            that holds the interval distance data for this vehicle.
        rules (EligibilityRules): The `EligibilityRules` that will be used
            to evaluate the initial charging eligibility if `start_soc` is
            less than 1.0.
        start_soc (float): The initial state of charge of the vehicle. Defaults
            to 1.0 (100%).
    """

    __slots__ = (
        'fleet_id',
        'powertrain',
        'max_soc',
        'evse_power',
        'distance_a',
        'fuel_burned_a',
        'deferred_a',
        'battery_a',
        'idx',
        # 'bracket',
        'begin_energy',
        'status',
        'soc_buffer'
    )

    def __init__(self, fleet_index: int, 
                 ptrain: Powertrain, distance: np.array,
                 rules: EligibilityRules, start_soc: float = 1.,
                 soc_buffer: float = 0):

        self.begin_energy = ptrain.energy_at_soc(start_soc)
        self.fleet_id = fleet_index
        self.powertrain = ptrain
        self.battery_a = np.zeros(distance.shape, dtype=np.float32)
        self.max_soc = 1 if ptrain.pev else 0
        self.evse_power = 0.
        self.distance_a = distance
        self.fuel_burned_a = np.zeros(len(distance), dtype=np.float32)
        self.soc_buffer = float(soc_buffer)
        self.deferred_a = np.zeros(len(distance), dtype=np.float32)

        self.idx = 0

        self.status = Status.DRIVING
        if distance[0] == 0.:
            if start_soc < 1.:
                self.status = stop_eligibility(0, distance, fleet_index, rules)
            else:
                self.status = Status.STOPPED

        # self.bracket = Bracket(0, fleet_index, distance_array, rules)

    @property
    def battery_state(self):
        """Provides the "incoming" battery state (from the previous index).

        Returns:
            A `float` representing the battery energy.
        """
        if self.idx > 0:
            return self.battery_a[self.idx - 1]
        return self.begin_energy

    @property
    def intervals(self):
        """The length of the distance array.
        
        Returns:
            A length of type `int`.
        
        Raises:
            `AssertionError` if the length is zero.
        """
        d = self.distance_a.shape[0]

        assert d > 0, \
            'Cannot use a zero-length drive profile.'
        return d

    @property
    def is_driving(self):
        """Returns a `bool` indicating if the vehicle is driving."""
        return self.distance_a[self.idx] > 0

    @property
    def is_stopped(self):
        """Returns the opposite of `is_driving`."""
        return not self.is_driving

    @property
    def is_new_stop(self):
        out = False
        if self.idx > 0:
            out = self.is_stopped and self.distance_a[self.idx - 1] > 0
        return out

    @property
    def will_drive_next(self):
        """Indicates whether the next interval will start a trip."""
        return self.is_stopped and self.distance_a[self.idx + 1] > 0

    @property
    def elapsed_distance(self):
        """Returns the elapsed distance for this simulation"""
        return np.sum(self.distance_a[:self.idx])

    @property
    def total_distance(self):
        """Returns the total distance for the simuilation.
        Note:
            This value will change if trips are deferred!
        """
        return np.sum(self.distance_a)
    
    @property
    def progress(self):
        """Returns how far through its distance array the vehicle is.
        The value is expressed as a `float` percentage wher 1.0 is 100%.
        """
        return (self.idx + 1) / self.intervals

    @property
    def distance(self):
        return self.distance_a[self.idx]

    @property
    def soc(self):
        """Returns the battery state of charge as a percentage."""
        out = 0
        if self.powertrain.batt_cap > 0:
            out = self.battery_a[self.idx - 1] / self.powertrain.batt_cap
        return out

    @property
    def ev_range(self):
        """Returns the range (in km) the battery has remaining."""
        return self.battery_a[self.idx - 1] * self.powertrain.ev_eff

    @property
    def max_ev_range(self):
        """The electric range the vehicle would have on a fully-charged
        battery.
        """
        return self.powertrain.batt_cap * self.max_soc * self.powertrain.ev_eff

    @property
    def evse_connected(self) -> bool:
        """Indicates if the attribute `evse_power` is greater than zero.
        A value of `True` indicates that an EVSE is connected to the vehicle.
        """
        return self.evse_power > 0.

    def index_sync(self, index: int):
        """Reports back how many intervals ahead or behind the vehicle is.

        Args:
            index (int): The index to check against

        Returns:
            An `int` indicating the differnce between `index` and the
                vehicle's index. A negative number indicates the vehicle is
                ahead. A positive number indicates it is behind. Zero
                indicates the indexes are in sync.
        """
        return index - self.idx

    def increment_index(self, index: int):
        """Advances the vehicle's index, but only if ithe vehicle is behind.

        Args:
            index (int): The index to attempt to advance to.

        Returns:
            An `int` indicating the number intervals behind or forward
                the vehicle's internal index is.

        """
        if index > self.idx:
            self.idx += 1
        return index - self.idx

    def set_battery(self, val: float, previous: bool = True):
        """Sets the battery energy.

        Args:
            val (float): The value to set the battery to.
                Nomalizes to be between 0 and the battery's max.
            previous (:obj: `bool`, optional): Whether to set the
                previous index's value. Defaults to `True`.
                If the current index is 0, then sets the beginning
                value for the vehicle.
        """
        true_val = max(0, min(self.powertrain.batt_cap, val))
        i = self.idx - 1 if previous else self.idx
        if i >= 0:
            self.battery_a[i] = true_val
        else:
            self.begin_energy = true_val

    def delta_battery(self, delta: float):
        """Sets the current index's battery state.

        The battery will be normalized to zero or the battery max.

        Args:
            delta (float): The change in the battery.
        Returns:
            A `float` representing the current battery energy.
        """
        val = self.battery_state + delta
        true_val = max(0., min(self.powertrain.batt_cap, val))
        self.battery_a[self.idx] = true_val
        return true_val

    def evse_compatible(self, dcfc: bool, plugs=None):

        if dcfc:
            if self.powertrain.dc_capable:
                if self.powertrain.dc_plug in plugs:
                    return True
            return False
        return True


    def connect_evse(self, power, dcfc, plugs=None):
        """Attempts to connect EVSE to the vehicle.

        Checks to see if the connection is compatible.

        Args:
            power (float): The input power from the EVSE
            dcfc (bool): Whether the input is DC.
            plugs (:obj: iterable, optional): An iterable
                containing the DC connectors available at
                this EVSE. Defaults to None.
                This is only needed for DC connections. AC
                connections are assumed to be J1772.

        Returns:
            `False` if connection fails, or a `float` representing
            the effective power level of the conenction (in kW).
        """

        if dcfc and self.powertrain.dc_power > 0.:
            # Check for DCFC plug compatibility
            if self.powertrain.dc_plug in plugs:
                self.evse_power = min(self.powertrain.dc_power, power)
                self.status = Status.CONNECTED
                return self.evse_power

        elif not dcfc and self.powertrain.ac_power > 0.:
            self.evse_power = min(self.powertrain.ac_power, power)
            self.status = Status.CONNECTED
            return self.evse_power

        return False

    def disconnect_evse(self):
        """Disconnects the EVSE and sets the power level to zero."""
        self.status = Status.DISCONNECTED
        self.evse_power = 0.

    def charge_battery(self, energy: float):
        """Charges the battery.

        Essentially a wrapper for `Vehicle.delta_battery()`,
        esxcept that this function checks to ensure there
        is EVSE connected.

        Args:
            energy (float): The energy to add

        Returns:
            A `float` representing net energy added.
                This adjusts for the fact the battery may
                be nearly full.
        """

        if self.powertrain.pev and self.evse_connected:
            new_state = self.delta_battery(energy)
            return self.battery_state - new_state
        return False

    def attempt_defer_trips(self, rules: EligibilityRules,
                            min_per_interval: float):
        
        if self.idx + 1 < self.distance_a.shape[0]:

            if (self.powertrain.bev and self.will_drive_next):

                acceptable_codes = (
                        Status.HOME_ELIGIBLE
                )

                max_len = self.distance_a.shape[0]
                
                target_index = next_trip(self.distance_a, self.idx)
                
                el_stop = None
                while el_stop is None and target_index < max_len:

                    stop = next_stop(self.distance_a, target_index)

                    code = stop_eligibility(
                        self.distance_a,
                        stop.end_index,
                        self.fleet_id,
                        rules,
                        use_mask=False
                    )

                    if code.begin_index == code.end_index:
                        target_index = max_len
                    elif code.code == Status.HOME_ELIGIBLE:
                        el_stop = code
                    else:
                        target_index = code.end_index
                                
                a, b = rewrite_deferred_trips(
                    self.distance_a,
                    self.deferred_a,
                    self.idx,
                    target_index,
                    self.ev_range,
                    self.powertrain.energy_at_soc(0.9),
                    self.evse_power * (min_per_interval/60.0),
                    self.soc_buffer
                )

                self.distance_a = a
                self.deferred_a = b

                # Recheck eligibility
                if self.powertrain.pev and self.status == Status.INELIGIBLE:
                    new_code = stop_eligibility(
                            self.distance_a,
                            self.idx,
                            self.fleet_id,
                            rules,
                            use_mask=False
                        )
                    self.status = new_code.code

    def advance_index(self, rules: EligibilityRules, min_per_interval: float):

        max_len = self.distance_a.shape[0]
        if self.idx < max_len:

            driving = self.distance_a[self.idx] > 0
            drive_next = False
            if self.idx < max_len - 1:
                drive_next = self.distance_a[self.idx + 1] > 0

            # Check for the end of a drive bracket
            if driving:
                code: ECode
                code = next_stop(self.distance_a, self.idx)
                try:
                    stop_index = drive_to_next_stop(
                        self.idx,
                        self.distance_a[:],
                        self.fuel_burned_a[:],
                        self.battery_a[:],
                        self.battery_state,
                        self.powertrain.ev_eff,
                        self.powertrain.ice_eff
                    )
                except AssertionError:
                    pass

                self.idx = stop_index
                code = stop_eligibility(
                    self.distance_a,
                    self.idx,
                    self.fleet_id,
                    rules
                )
                self.status = code.code

            elif drive_next:
                self.attempt_defer_trips(rules, min_per_interval)
                self.idx += 1
                if self.idx < max_len:
                    self.battery_a[self.idx] = self.battery_a[self.idx - 1]
            else:
                self.idx += 1
                if self.idx < max_len:
                    self.battery_a[self.idx] = self.battery_a[self.idx - 1]
