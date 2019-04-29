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
    home_threshold_min: float = 300.0
    away_threshold_min: float = 90.0
    soc_deferment_buffer: float = .2
    idle_load_kw: float =  0.
    home_mask_rules: dict = [
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
    home_banks = [
        {
            'probability': 1.,
            'queue': QueueMode.DEFAULT,
            'evse': [
                (EVSEType(7), 0, 0, .5)
            ]
        }
    ]

    away_banks = [
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

    def make_home_banks(self, fleet_size):
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

