
import math
from typing import NamedTuple

from .vehicle import Vehicle
# from evnrg.simulation.eligibility import EligibilityRules

__all__ = [
    'EVSE',
    'EVSEType'
]

class EVSEType(NamedTuple):
    max_power: float = 0.
    dc: bool = False
    max_soc: float = 1.
    dc_plugs: tuple = (None,)
    v2g_buffer: float = 0.20  # Arbitrage of 20% max SoC
    v2g_capable: bool = False


class EVSE(object):

    def __init__(self, model: EVSEType = EVSEType()):
        self.model = model
        self.power = model.max_power
        self.throttle = 1.
        self.vehicle = None
        self.target_soc = model.max_soc
        self.v2g_mode = False

    # TODO: Create properties that show the maximum and minimum SoCs and
    # energy available for V2G use.
    # Can also create methods to describe temporally-based V2G potential
    # For the purpose ofsafely bidding into an ISO/utility's grid services,
    # we could estimate this as:
    # (time_until_departure / 2) * charge_power

    @property
    def is_connected(self):
        return self.vehicle is not None
    
    @property
    def is_available(self):
        return self.vehicle is None

    @property
    def demand(self):
        out = 0
        if self.is_connected:        
            if self.v2g_mode:
                # TODO: If the vehicle is discharging in V2G mode
                pass
            else:
                out = self.power * self.throttle
        return out

    @property
    def charge_completed(self):
        out = False
        if self.vehicle is not None:
            out = self.vehicle.soc >= self.target_soc
        return out

    def charge_vehicle(self, minutes: float):
        true_power = 0
        if self.is_connected:
            if not self.charge_completed:
                power = self.power * self.throttle
                energy_max = self.vehicle.powertrain.energy_at_soc(self.target_soc)
                energy_potential = power * (minutes / 60.0)
                energy_limit = energy_max - self.vehicle.battery_state
                energy_to_add = min(energy_potential, energy_limit)
                self.vehicle.charge_battery(energy_to_add)
                true_power = energy_to_add / (minutes / 60.0)
        return true_power

    def connect_vehicle(self, v: Vehicle):
        out = False
        target = min(self.target_soc, v.max_soc)
        if target > v.soc:
            if v.connect_evse(self.power, self.model.dc, self.model.dc_plugs):
                self.target_soc = target
                power = 0.
                if self.model.dc and v.powertrain.dc_power > 0.:
                    power = min(self.model.max_power, v.powertrain.dc_power)
                else:
                    power = min(self.model.max_power, v.powertrain.ac_power)
                self.power = power
                self.vehicle = v
                out = True
        return out

    def intervals_to_target_soc(self, minutes_per_interval: float):
        out = 0
        if self.vehicle:
            current_energy = self.vehicle.battery
            target_energy = self.vehicle.energy.energy_at_soc(self.target_soc)

            energy_needed = target_energy - current_energy

            hours_required = energy_needed / self.power

            out = math.ceil(hours_required / minutes_per_interval)
        return out

    def connect_vehicle_soc_target(self, v: Vehicle, desired_soc: float):
        self.target_soc = desired_soc
        return self.connect_vehicle(v)

    def disconnect_vehicle(self):
        out = None
        if self.is_connected:
            v = self.vehicle
            self.vehicle.disconnect_evse()
            self.vehicle = None
            self.target_soc = self.model.max_soc
            out = v
        return out
