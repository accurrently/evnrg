import enum
import random
import math
import copy
from typing import List

import numpy as np

from .evse import EVSE
from .vehicle import Vehicle
from .status import Status

__all__ = [
    'QueueMode',
    'Bank'
]


class QueueMode(enum.IntEnum):

    DEFAULT = enum.auto()
    STACK = enum.auto()  # Last-in, first out
    RANDOM = enum.auto()
    SOC = enum.auto()
    TIME = enum.auto()
    RSOC = enum.auto()
    RTIME = enum.auto()

    @classmethod
    def lookup(cls, s: str):
        return {
            'default': QueueMode.DEFAULT,
            'stack': QueueMode.STACK,
            'random': QueueMode.RANDOM,
            'soc': QueueMode.SOC,
            'time': QueueMode.TIME,
            'rsoc': QueueMode.RSOC,
            'rtime': QueueMode.RTIME
        }.get(s, QueueMode.DEFAULT)


ALLOWED_QUEUE_MODES = {
    QueueMode.DEFAULT,
    QueueMode.STACK,
    QueueMode.RANDOM,
    QueueMode.SOC,
    QueueMode.TIME,
    QueueMode.RSOC,
    QueueMode.RTIME
}


class Bank(object):

    __slots__ = (
        'max_power',
        'capacity',
        'queue_probability',
        'dynamic_size',
        'queue_mode',
        'queue',
        'available_evse',
        'occupied_evse',
        'demand_profile',
        'occupancy_profile'
    )

    def __init__(self, max_power: float = 0., capacity: float = 0., queue_probability: float = 1.,
                 dynamic_size: bool = False, queue_mode: QueueMode = QueueMode.DEFAULT,
                 evse: list = [], demand_profile: np.array = np.empty(0),
                 occupancy_profile: np.array = np.empty(0)):
        
        self.max_power = max_power
        self.capacity = capacity
        self.queue_probability = queue_probability
        self.dynamic_size = dynamic_size
        self.queue_mode = queue_mode
        self.queue = []
        self.available_evse = evse
        self.occupied_evse = []
        self.demand_profile = demand_profile
        self.occupancy_profile = occupancy_profile

    @property
    def num_available(self):
        return len(self.available_evse)

    @property
    def occupancy(self) -> float:
        if self.size > 0:
            return float(len(self.occupied_evse) / self.size)
        return 0

    @property
    def num_occupied(self):
        return len(self.occupied_evse)
    
    @property
    def size(self):
        return self.num_available + self.num_occupied

    @property
    def num_operating(self) -> int:
        x = 0
        for evse in self.occupied_evse:
            if evse.is_connected:
                x += 1
        return x

    @property
    def total_demand(self) -> float:
        x = 0.0
        for evse in self.occupied_evse:
            if evse.is_connected:
                x += evse.demand
        return x

    @property
    def pct_capacity(self):
        if self.capacity > 0:
            return self.total_demand / self.capacity
        return 0

    def record_demand(self, idx: int):
        self.demand_profile[idx] = self.total_demand
    
    def record_occupancy(self, idx: int):
        self.occupancy_profile[idx] = self.num_occupied

    def add_evse_with_rules(self, fleet_size: int,
                            chargers: List[EVSE], rules: dict):
        for evse, rule in zip(chargers, rules):
            min_num = rule.get('minimum', 1)
            max_num = rule.get('maximum', 0)
            ratio = rule.get('evse_per_vehicle', 0)

            num_to_add = max(min_num, fleet_size*ratio)

            if max_num > 0:
                num_to_add = min(num_to_add, max_num)

            num_to_add = math.ceil(num_to_add)

            for i in range(num_to_add):
                self.available_evse.append(copy.deepcopy(evse))
                if self.dynamic_size:
                    break

            if self.dynamic_size:
                break

    def charge_connected(self, minutes_per_interval: float, idx: int):
        demand = 0.
        occ = 0
        for evse in self.occupied_evse:
            if evse.vehicle.idx <= idx:
                demand += evse.charge_vehicle(minutes_per_interval)
                occ += 1
        self.demand_profile[idx] = demand
        self.occupancy_profile[idx] = (occ / self.size) if self.size > 0 else 0

    def set_profile_length(self, size: int):
        self.demand_profile = np.zeros(size, dtype=np.float32)
        self.occupancy_profile = np.zeros(size, dtype=np.uint8)

    def add_evse(self, evse: EVSE):
        self.available_evse.append(evse)
        self.size = min(255, self.size + 1)
        self.capacity += evse.max_power

    def remove_evse(self, evse: EVSE):
        out = True
        try:
            self.capacity = max(0, self.capacity - evse.max_power)
            self.available_evse.remove(evse)
            self.size = max(0, self.size - 1)
        except Exception:
            out = False
        return out

    def enqueue_vehicle(self, vic: Vehicle):
        if vic.powertrain.pev:
            vic.status = Status.IN_QUEUE
            self.queue.append(vic)

    def enqueue_vehicle_prob(self, vic: Vehicle):
        out = False
        if vic.powertrain.pev:
            if random.random() <= self.queue_probability:
                self.enqueue_vehicle(vic)
                out = True
        return out

    def dequeue_vehicle(self, vic: Vehicle):
        out = False
        try:
            self.queue.remove(vic)
            out = True
        except Exception:
            out = False
        return out

    def dequeue_vehicle_index(self, fleet_index: int):
        out = None
        for v in self.queue:
            if v.fleet_index == fleet_index:
                if self.dequeue_vehicle(v):
                    out = v
        return out

    def next_available_evse(self):
        out = None
        if self.available_evse:
            if not self.available_evse[0].is_connected:
                out = self.available_evse[0]
        return out

    def dequeue_early_departures(self, idx: int):
        departing = []
        for v in self.queue:
            if v.idx <= idx:
                if v.is_driving:
                    if self.dequeue_vehicle(v):
                        departing.append(v)
        return departing

    def pop_next_vehicle(self):
        winner = None
        if self.queue:
            # Default queue mode
            winner = self.queue[0]

            # Stack
            if self.queue_mode == QueueMode.STACK:
                winner = self.queue[-1]

            # Random
            elif self.queue_mode == QueueMode.RANDOM:
                winner = self.queue[random.randrange(len(self.queue))]

            # SoC (lowest first)
            elif self.queue_mode == QueueMode.SOC:
                for v in self.queue[1:]:
                    if v.soc < winner.soc:
                        winner = v

            # SoC (Highest first)
            elif self.queue_mode == QueueMode.RSOC:
                for v in self.queue[1:]:
                    if v.soc > winner.soc:
                        winner = v

            # Time to charge (assume 6kW power)
            elif self.queue_mode == QueueMode.TIME:
                for v in self.queue[1:]:
                    v_time = v.minutes_to_charged(6, 1)
                    winner_time = winner.minutes_to_charged(6, 1)
                    if v_time > winner_time:
                        winner = v

            # Lowest time to charge
            elif self.queue_mode == QueueMode.RTIME:
                for v in self.queue[1:]:
                    v_time = v.minutes_to_charged(6, 1)
                    winner_time = winner.minutes_to_charged(6, 1)
                    if v_time < winner_time:
                        winner = v

            if not self.dequeue_vehicle(winner):
                winner = None
        return winner

    def process_queue(self):
        """Checks each EVSE in the bank.
        If the EVSE is not connected to a vehicle, dequeue one and connect it.
        """
        num_connected = 0

        # Only do work if there are available EVSE
        if self.num_available > 0:

            # If we're dynamically sized, conect every Vehicle
            # that is compatible with the type of charger.
            # If a Vehicle is ineligible (incompatible or the SoC is too high),
            # remove it from the queue
            if self.dynamic_size:
                while self.queue:
                    vehicle = self.pop_next_vehicle()
                    evse: EVSE
                    evse = self.available_evse[0]
                    if vehicle.evse_compatible(evse.model.dc, evse.model.dc_plugs):
                        evse = copy.deepcopy(self.available_evse[0])
                        if evse.connect_vehicle(vehicle):
                            self.occupied_evse.append(evse)
                            num_connected += 1

            # Otherwise, process a queue normally
            else:
                ineligible_vehicles = []

                # As long as there is EVSE available and there are still
                # vehicles in the queue
                while self.num_available > 0 and self.queue:
                    vehicle = self.pop_next_vehicle()
                    
                    # Check for a valid vehicle
                    if vehicle is not None:
                        connected = False
                        for evse in self.available_evse:
                            # Check if the EVSE can accept connections
                            # and the EVSE is compatible with the Vehicle and 
                            # its needs
                            if evse.connect_vehicle(vehicle):
                                num_connected += 1      # Update our count

                                # Move the EVSE to the list of occupied
                                self.available_evse.remove(evse)
                                self.occupied_evse.append(evse)

                                # Remember that we connected this one
                                connected = True
                                # Break out of the loop;
                                # No need to look at other EVSE
                                break

                        # If we failed to connect, the vehicle is ineligible
                        # for any of the available EVSE.
                        if not connected:
                            ineligible_vehicles.append(vehicle)

                # Now that we're done popping Vehicles,
                # requeue the ineligible ones so they can wait for EVSE that
                # will meet their needs

                # First, check if we're using a stack, so we can re-stack the
                # queue
                if self.queue_mode == QueueMode.STACK:
                    ineligible_vehicles = ineligible_vehicles[::-1]

                for v in ineligible_vehicles:
                    self.queue.append(v)

        # Return the number of vehicles that got to connect
        return num_connected

    def disconnect_completed_vehicles(self, idx: int):
        """
        Disconnects any vehicles that are effectively done charging.
        """
        num_disconnected = 0
        for evse in self.occupied_evse:
            if evse.is_connected and not evse.v2g_mode:
                vehicle = evse.vehicle
                if vehicle.idx <= idx:
                    if evse.charge_completed:
                        if evse.disconnect_vehicle() is not None:
                            num_disconnected += 1
                            self.occupied_evse.remove(evse)
                            if not self.dynamic_size:
                                self.available_evse.append(evse)
        return num_disconnected

    def disconnect_departing_vehicles(self, idx: int):
        num_disconnected = 0
        for evse in self.occupied_evse:
            if evse.is_connected:
                vehicle = evse.vehicle
                if vehicle.idx <= idx:
                    if evse.vehicle.is_driving:
                        v = evse.vehicle
                        if evse.disconnect_vehicle() is not None:
                            v.status = Status.DRIVING
                            num_disconnected += 1
                            self.occupied_evse.remove(evse)
                            if not self.dynamic_size:
                                self.available_evse.append(evse)
        return num_disconnected
