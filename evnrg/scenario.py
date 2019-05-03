import numpy as np
import math

from typing import NamedTuple, List
import uuid

from .powertrain import Powertrain
from .evse import EVSEType
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
                EVSEType(max_power=7., dc=False, min_=1, max_=0, pro_=.5)
            ]
        }
    ]
    away_banks: list = [
        {
            'probability': .1,
            'evse': [
                EVSEType(
                    max_power=50.,
                    dc=True,
                    max_soc=.8,
                    dc_plugs=(DCPlug.CHADEMO, DCPlug.COMBO),
                    min_=1,
                    max_=0,
                    pro_=0
                )
            ]
        },
        {
            'probability': .2,
            'evse': [
                EVSEType(max_power=7., dc=False, min_=1, max_=0, pro_=0)
            ]
        }
    ]

    

