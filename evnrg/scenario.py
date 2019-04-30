import numpy as np
import math

from typing import NamedTuple, List
import uuid

from .powertrain import Powertrain
from .evse import EVSEType, EVSE
from .plug import DCPlug
from .bank import QueueMode, Bank


class Scenario(NamedTuple):
    powertrains: List[Powertrain]
    distribution: list
    run_id: str = uuid.uuid4().hex
    home_threshold_min: float = 180.0 # Default to 3 hours
    away_threshold_min: float = 90.0 # default to 1.5 hours
    soc_deferment_buffer: float = .2 # default to 20% SoC buffer
    idle_load_kw: float =  0.
    home_mask_rules: list = [
        # An example of a time-based mask
        {
            'type': 'time',
            'begin': '19:00',
            'end': '08:00'
        }
        # You could feasibly also use a location-based
        #{
            # Type
        #    'type': 'location',
            # Some lat
        #    'latitude': 12.5987653,
            # some long
        #    'longitude': 13.908643,
            # Some radius in meters
        #    'radius': 400
        #}
    ]
    home_banks: list = [
        {
            'probability': 1.,
            'queue': QueueMode.DEFAULT,
            'evse': [
                (EVSEType(7), 0, 0, .5)
            ]
        }
    ]
    away_banks: list = [
        {
            'probability': .1,
            'evse': [
                EVSEType(50, True, .8, (DCPlug.CHADEMO, DCPlug.COMBO))
            ]
        },
        {
            'probability': .2,
            'evse': [
                EVSEType(7.0)
            ]
        }
    ]

    def make_home_banks(self, fleet_size, rows):
        home_banks = []
        for bank_info in self.home_banks:
            evse_list = []
            for evse_type, nmin, nmax, proportion in bank_info.get('evse'):
                n_evse = min(nmin, max(nmax, math.floor(proportion * fleet_size)))
                for i in range(n_evse):
                    evse_list.append(EVSE(evse_type))
            bank = Bank(
                max_power=bank_info.get('max_power', 0.),
                capacity=bank_info.get('capacity', 0.),
                evse=evse_list,
                queue_probability=bank_info.get('probability', 1.),
                queue_mode=bank_info.get('queue', QueueMode.DEFAULT),
                demand_profile=np.zeros(rows, dtype=np.float32),
                occupancy_profile=np.zeros(rows, dtype=np.uint8),
                dynamic_size=False
            )
            home_banks.append(bank)
        return home_banks
    
    def make_away_banks(self, rows):
        away_banks = []
        for bank_info in self.away_banks:
            evse_list = []
            for evse_type in bank_info.get('evse'):
                evse_list.append(EVSE(evse_type))
            demand_a = np.zeros(rows, dtype=np.float32)
            occupancy_a = np.zeros(rows, dtype=np.uint8)
            bank = Bank(
                max_power=bank_info.get('max_power', 0.),
                capacity=bank_info.get('capacity', 0.),
                evse=evse_list,
                queue_probability=bank_info.get('probability', .2),
                queue_mode=bank_info.get('queue', QueueMode.DEFAULT),
                demand_profile=np.zeros(rows, dtype=np.float32),
                occupancy_profile=np.zeros(rows, dtype=np.uint8),
                dynamic_size=True
            )
            away_banks.append(bank)
        return away_banks

