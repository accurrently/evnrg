from typing import NamedTuple, List
import uuid

from .powertrain import Powertrain
from .evse import EVSEType
from .plug import DCPlug
from .bank import QueueMode


class Scenario(NamedTuple):
    powertrains: List[Powertrain]
    distribution: list
    run_id: str = uuid.uuid4().hex
    home_threshold_min: float = 300.0
    away_threshold_min: float = 120.0
    soc_deferment_buffer: float = .2
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
                EVSEType(7),
                EVSEType(7),
                EVSEType(6),
                EVSEType(6)
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
