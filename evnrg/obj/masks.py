from typing import NamedTuple

import pandas as pd
import numpy as np

class HomeMask(NamedTuple):

    @classmethod
    def empty(cls, df: pd.DataFrame):
        return pd.DataFrame(
            data = np.zeros(
                df.values.shape,
                dtype = np.bool_
            ),
            index = df.index
        )

    def get_mask(self, df: pd.DataFrame):
        return HomeMask.empty(df)
        

class TimeMask(HomeMask):
    begin: str = '19:00'
    end: str = '08:00'

    def get_mask(self, df: pd.DataFrame):
        mask = HomeMask.empty(df)

        mask[df.between_time(self.begin, self.end).index] = True

        return mask

# TODO: Implement LocationMask logic
class LocationMask(HomeMask):
    latitude: float
    longitude: float
    radius: float
    lat_df: pd.DataFrame
    lon_df: pd.DataFrame

    def get_mask(self, df: pd.DataFrame):

        return HomeMask.empty(df)

class MaskRules(NamedTuple):

    masks: list
    
    def make_mask(self, df: pd.DataFrame):
        out = HomeMask.empty(df)

        mask: HomeMask
        for mask in self.masks:
            out = out | mask.get_mask(df)
        
        return out


