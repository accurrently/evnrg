
import math
from typing import NamedTuple

# from .vehicle import Vehicle
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
    min_: int = 0
    max_: int = 0
    pro_: float = 0.
    #v2g_buffer: float = 0.20  # Arbitrage of 20% max SoC
    #v2g_capable: bool = False

