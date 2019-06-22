
import math
from typing import NamedTuple, List
import numpy as np
import numba as nb

from .plug import DCPlug

# from .vehicle import Vehicle
# from evnrg.simulation.eligibility import EligibilityRules

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

class EVSEDef(NamedTuple):
    evse_type: EVSEType
    minimum: int = 0
    maximum: int = 0
    proportion: float = 0.

    @property
    def max_power(self):
        return self.evse_type.max_power
    
    @property
    def dc(self):
        return self.evse_type.dc
    
    @property
    def max_soc(self):
        return self.evse_type.max_soc
    
    @property
    def dc_plugs(self):
        return self.evse_type.dc_plugs

evse_ = np.dtype([
    ('bank_id', np.uint64, 1),
    ('power_max', np.float32, 1),
    ('bank_power_max', np.float32, 1),
    ('power', np.float32, 1),
    ('max_soc', np.float32, 1),
    ('probability', np.float32, 1),
    ('dc', np.bool_, 1),
    ('plug_chademo', np.bool_, 1),
    ('plug_combo', np.bool_, 1),
    ('plug_tesla', np.bool_, 1)
])

nb_evse_ = nb.from_dtype(evse_) 

class Bank(object):

    __slots__ = (
        'evse',
        'probability',
        'max_power',
    )

    def __init__(self, probability: float = 1., max_power: float = -1.):
        self.probability = probability
        self.evse = []
        self.max_power = max_power

    def add_evse_def(self, evse_def: EVSEDef):

        self.evse.append(evse_def)
    
    def add_evse(self, e: EVSEType, minimum: float = 0., maximum: float = 0., prop: float = 1.):
        self.evse.append(
            EVSEDef(e, minimum, maximum, prop)
        )
    
    def make_bank(self, fleet_size: int, bank_id: int, away: bool = False):
        power_max = []
        bank_power_max = []
        max_soc = []
        probability = []
        dc = []
        plug_chademo = []
        plug_combo = []
        plug_tesla = []

        bank_running_power = 0
        n_evse = 0

        e: EVSEDef
        for e in self.evse:
            copies = round(fleet_size * e.proportion)
            if e.max_ > 0:
                copies = int(min(e.maximum, copies))
            copies = int(max(e.minimum, copies))

            power_max.extend([e.max_power] * copies)
            max_soc.extend([e.max_soc] * copies)
            dc.extend([e.dc] * copies)
            plug_chademo.extend([bool(DCPlug.CHADEMO in e.dc_plugs)] * copies)
            plug_combo.extend([bool(DCPlug.COMBO in e.dc_plugs)] * copies)
            plug_tesla.extend([bool(DCPlug.TESLA in e.dc_plugs)] * copies)
            n_evse += copies
            bank_running_power += e.max_power * copies

        bank_power_max_val = self.max_power if self.max_power > 0 else bank_running_power
        bank_power_max.extend([bank_power_max_val for i in range(n_evse)])
    
        a = np.empty(n_evse, dtype=evse_)

        a[:]['bank_id'] = bank_id
        a[:]['power'] = 0.
        a[:]['power_max'] = power_max
        a[:]['bank_power_max'] = bank_power_max
        a[:]['max_soc'] = max_soc
        a[:]['probability'] = self.probability
        a[:]['dc'] = dc
        a[:]['plug_chademo'] = plug_chademo
        a[:]['plug_combo'] = plug_combo
        a[:]['plug_tesla'] = plug_tesla

        if away:
            a[:]['power'] = a[:]['power_max']

        return a

    @classmethod
    def make_banks(cls, banks: List[Bank], fleet_size: int, away: bool = False):

        out = np.empty(0, dtype=evse_)
        bank_id = 0
        for bank in banks:
            b = bank.make_bank(fleet_size, bank_id, away)
            out = np.append(b)
            bank_id += 1
        return out


def make_np_evse_banks(evse_banks: List[Bank], fleet_size: int, away=False):

    n_evse = 0

    bank_ids = []
    power_max = []
    bank_power_max = []
    max_soc = []
    probability = []
    dc = []
    plug_chademo = []
    plug_combo = []
    plug_tesla = []  

    bank_id = 0
    for bank in evse_banks:
        
        bank_running_power = 0
        n_in_bank = 0
        e: EVSEDef
        for e in bank.evse:
            copies = round(fleet_size * e.proportion)
            if e.max_ > 0:
                copies = int(min(e.maximum, copies))
            copies = int(max(e.minimum, copies))

            for i in range(copies):
                bank_ids.append(bank_id),
                power_max.append(e.max_power)
                
                max_soc.append(e.max_soc)
                probability.append(bank.probability)
                dc.append(e.dc)
                plug_chademo.append(bool(DCPlug.CHADEMO in e.dc_plugs))
                plug_combo.append(bool(DCPlug.COMBO in e.dc_plugs))
                plug_tesla.append(bool(DCPlug.TESLA in e.dc_plugs))
                n_evse += 1
                bank_running_power += e.max_power
                n_in_bank += 1
        bank_id += 1
        bank_power_max_val = bank.max_power if bank.max_power > 0 else bank_running_power
        bank_power_max.extend([bank_power_max_val for i in range(n_in_bank)])
    
    a = np.empty(n_evse, dtype=evse_)

    a[:]['bank_id'] = bank_ids
    a[:]['power'] = 0.
    a[:]['power_max'] = power_max
    a[:]['bank_power_max'] = bank_power_max
    a[:]['max_soc'] = max_soc
    a[:]['probability'] = probability
    a[:]['dc'] = dc
    a[:]['plug_chademo'] = plug_chademo
    a[:]['plug_combo'] = plug_combo
    a[:]['plug_tesla'] = plug_tesla

    if away:
        a[:]['power'] = a[:]['power_max']

    return a





