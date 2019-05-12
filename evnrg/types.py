import numpy as np
import numba as nb
import pandas as pd

from .evse import EVSEType
from .powertrain import Powertrain, PType
from .plug import DCPlug

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

def make_evse_banks(evse_banks: list, fleet_size: int, away=False):

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
        e: EVSEType
        for e in bank['evse']:
            copies = round(fleet_size * e.pro_)
            if e.max_ > 0:
                copies = int(min(e.max_, copies))
            copies = int(max(e.min_, copies))

            for i in range(copies):
                bank_ids.append(bank_id),
                power_max.append(e.max_power)
                
                max_soc.append(e.max_soc)
                probability.append(bank.get('probability', 1.))
                dc.append(e.dc)
                plug_chademo.append(bool(DCPlug.CHADEMO in e.dc_plugs))
                plug_combo.append(bool(DCPlug.COMBO in e.dc_plugs))
                plug_tesla.append(bool(DCPlug.TESLA in e.dc_plugs))
                n_evse += 1
                bank_running_power += e.max_power
                n_in_bank += 1
        bank_id += 1
        bank_power_max_val = bank.get('max_power', bank_running_power)
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


vehicle_ = np.dtype([
    ('type', np.int32, 1),
    ('ice_eff', np.float32, 1),
    ('ice_gal_kwh', np.float32, 1),
    ('fuel_co2e', np.float32, 1),
    ('ev_eff', np.float32, 1),
    ('ev_max_batt', np.float32, 1),
    ('ac_max', np.float32, 1),
    ('dc_max', np.float32, 1),
    ('dc_plug', np.int32, 1),
    ('home_evse_id', np.int64, 1),
    ('away_evse_id', np.int64, 1),
    ('input_power', np.float32, 1),
    ('input_max_soc', np.float32, 1)
])

nb_vehicle_ = nb.from_dtype(vehicle_)

def make_fleet(
    vtypes: list,
    ice_effs: list,
    ice_gal_kwhs: list,
    ev_effs: list,
    ev_max_batts: list,
    ac_maxes: list, 
    dc_maxes: list,
    dcplugs: list,
    co2e: list):

    n = len(vtypes)
    a = np.empty(n, dtype=vehicle_)


    a[:]['type'] = vtypes
    a[:]['ice_eff'] = ice_effs
    a[:]['ice_gal_kwh'] = ice_gal_kwhs
    a[:]['ev_eff'] = ev_effs
    a[:]['ev_max_batt'] = ev_max_batts
    a[:]['ac_max'] = ac_maxes
    a[:]['dc_max'] = dc_maxes
    a[:]['dc_plug'] = dcplugs
    a[:]['home_evse_id'] = -1
    a[:]['away_evse_id'] = -1
    a[:]['input_power'] = 0.
    a[:]['input_max_soc'] = 0.
    a[:]['fuel_co2e'] = co2e
    
    return a


