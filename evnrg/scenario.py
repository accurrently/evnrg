import numpy as np
import math
import enum

from typing import NamedTuple, List
import uuid

from .powertrain import Powertrain
from .evse import EVSEType
from .plug import DCPlug
from .fleet import FleetDef
from .masks import MaskRules

class Scenario(NamedTuple):
    # powertrains: List[Powertrain]
    # distribution: list
    fleet_def: FleetDef
    mask_rules: MaskRules
    name: str = uuid.uuid4().hex
    home_threshold_min: float = 300.0 # Default to 5 hours
    away_threshold_min: float = 120.0 # default to 2 hours
    soc_buffer: float = .2 # default to 20% SoC buffer
    interval_min: float = 5
    idle_load_kw: float =  0.
    home_banks: list = []
    away_banks: list = []
    

