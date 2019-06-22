from typing import NamedTuple

import numpy as np
import pandas as pd

from .powertrain import Powertrain

vehicle_ = np.dtype([
    ('type', np.int32, 1),
    ('ice_eff', np.float32, 1),
    ('ice_gal_kwh', np.float32, 1),
    ('fuel_co2e', np.float32, 1),
    ('fuel_mj_gal', np.float32, 1),
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

class FleetDef(NamedTuple):
    powertrains: list = []
    distribution: list = []

    def make_fleet(self, size: int):

        if len(self.powertrains) <= 0:
            raise ValueError('Must have at least one Powertrain')
        
        # By default, assign all to the first powertrain
        assignments = [0] * size

        ntrains = len(self.powertrains)
        ndist = len(self.distribution)

        distribution = self.distribution
        dist_sum = sum(distribution)

        if dist_sum > 1:
            raise ValueError('Distribution must sum to <= 1.')
        
        

        pt_ids = range(ntrains)

        # Explicit assignment
        if ndist == 0:
            assignments = [pt_ids[i % ntrains] for i in range(size)]
        
        else:

            if ndist < ntrains:
                dist_fill = ntrains - ndist
                dist_remain = (1 - dist_sum) / dist_fill
                distribution.extend([dist_remain] * dist_fill)
            elif ndist > ntrains:
                j = (1 - sum(distribution[:ntrains])) / ntrains
                distribution = [distribution[i] + j for i in range(ntrains)]
            
            
            assignments = np.random.choice(
                range(ntrains),
                size=size,
                replace=True,
                p=np.array(distribution)
            )
        
        a = np.empty(size, dtype=vehicle_)
        
        a[:]['type'] = [int(self.powertrains[i].ptype) for i in assignments]
        a[:]['ice_eff'] = [self.powertrains[i].ice_eff for i in assignments]
        a[:]['ice_gal_kwh'] = [self.powertrains[i].ice_gal_kwh for i in assignments]
        a[:]['ev_eff'] = [self.powertrains[i].ev_eff for i in assignments]
        a[:]['ev_max_batt'] = [self.powertrains[i].batt_cap for i in assignments]
        a[:]['ac_max'] = [self.powertrains[i].ac_power for i in assignments]
        a[:]['dc_max'] = [self.powertrains[i].dc_power for i in assignments]
        a[:]['dc_plug'] = [self.powertrains[i].dc_plug for i in assignments]
        a[:]['home_evse_id'] = -1
        a[:]['away_evse_id'] = -1
        a[:]['input_power'] = 0.
        a[:]['input_max_soc'] = 0.
        a[:]['fuel_co2e'] = [(self.powertrains[i].fuel.kgCO2_gal if self.powertrains[i].fuel else 0.) for i in assignments]
        a[:]['fuel_mj_gal'] = [(self.powertrains[i].fuel.MJ_gal if self.powertrains[i].fuel else 0.) for i in assignments]
        
        return a

