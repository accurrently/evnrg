import numpy as np
import pandas as pd

from .vehicle import Vehicle
from .powertrain import Powertrain
from .eligibility import EligibilityRules

__all__ = [
    'Fleet'
]

class Fleet(object):
    """A collection of vehicles and data connecting them."""

    __slots__ = (
        'df',
        'vehicles',
        'size'
    )

    def __init__(self, df: pd.DataFrame, powertrains: list,
                 probabilities: list, rules: EligibilityRules):
        self.df = df.astype(np.float32, copy=False)
        self.vehicles = []
        self.size = self.df.values.shape[1]

        vnames = df.columns

        if isinstance(powertrains, list):
            for pt in powertrains:
                if not isinstance(pt, Powertrain):
                    raise TypeError()

            trains = powertrains
            num_trains = len(powertrains)
            num_probs = len(probabilities)
            probs = probabilities
            if not probabilities:
                probs = [float(1.0 / num_trains) for i in range(num_trains)]
            elif num_probs < num_trains:
                current_sum = sum(probs)
                if current_sum < 1:
                    remaining = num_trains - num_probs
                    distribution = (1 - current_sum) / remaining
                    probs.extend([distribution for i in range(remaining)])
                else:
                    trains = powertrains[:num_probs]
            elif num_trains < num_probs:
                probs = probabilities[:num_trains]

            pt_assignments = np.random.choice(
                range(num_trains),
                size=self.size,
                replace=True,
                p=np.array(probs)
            )

            for pt, vid, vname in zip(pt_assignments, range(self.size), vnames):
                vehicle = Vehicle(
                    vid,
                    trains[pt],
                    self.df.values[:, vid],
                    rules,
                    vid = vname
                )
                self.vehicles.append(vehicle)

    @property
    def vehicle_ids(self):
        return self.df.columns.to_list()

    @property
    def fleet_ids(self):
        return [self.df.columns.get_loc(i) for i in self.df.columns]

    def add_vehicle(self, col_id, pt: Powertrain, rules: EligibilityRules):
        if not (col_id in self.vehicle_ids):
            raise IndexError()

        v = Vehicle(
            list(self.df.columns).index[col_id],
            self.df[col_id].values,
            rules
        )
        self.vehicles.append(v)

    def assign_powertrains(self, powertrains: list, 
                           probabilities: list, rules: EligibilityRules):

        self.vehicles = []

        if isinstance(powertrains, list):
            for pt in powertrains:
                if not isinstance(pt, Powertrain):
                    raise TypeError()

            trains = powertrains
            num_trains = len(powertrains)
            num_probs = len(probabilities)
            probs = probabilities
            if not probabilities:
                probs = [float(1.0 / num_trains) for i in range(num_trains)]
            elif num_probs < num_trains:
                current_sum = sum(probs)
                if current_sum < 1:
                    remaining = num_trains - num_probs
                    distribution = (1 - current_sum) / remaining
                    probs.extend([distribution for i in range(remaining)])
                else:
                    trains = powertrains[:num_probs]
            elif num_trains < num_probs:
                probs = probabilities[:num_trains]

            pt_assignments = np.random.choice(trains, self.size, True, probs)

            for pt, vid in zip(pt_assignments, self.vehicle_ids):
                self.add_vehicle(vid, pt)

    def get_powertrain_distribution(self, powertrains: list,
                                    probabilities: list,
                                    rules: EligibilityRules):

        self.vehicles = []

        if isinstance(powertrains, list):
            for pt in powertrains:
                if not isinstance(pt, Powertrain):
                    raise TypeError()

            trains = powertrains
            num_trains = len(powertrains)
            num_probs = len(probabilities)
            probs = probabilities
            if not probabilities:
                probs = [float(1.0 / num_trains) for i in range(num_trains)]
            elif num_probs < num_trains:
                current_sum = sum(probs)
                if current_sum < 1:
                    remaining = num_trains - num_probs
                    distribution = (1 - current_sum) / remaining
                    probs.extend([distribution for i in range(remaining)])
                else:
                    trains = powertrains[:num_probs]
            elif num_trains < num_probs:
                probs = probabilities[:num_trains]        

            pt_assignments = np.random.choice(trains, self.size, True, probs)

            for pt, vid in zip(pt_assignments, self.vehicle_ids):
                self.add_vehicle(vid, pt)



        
        
        
