from typing import NamedTuple
import datetime

import pandas as pd
import numba as nb
import numpy as np
import dask
import dask.dataframe as dd
import dask.array as da
from dask_ml.cluster import KMeans

from geopy import distance




class PremaskRule(NamedTuple):

    def genmask(self, ddf: dask.dataframe.DataFrame):
        # Return a Dask Series of False values
        return ddf.map_partitions(
            lambda df: df.apply(lambda x: False, axis=1),
            meta=('base_mask', 'bool')
        )

class GeoPremaskRule(NamedTuple):
    dwell_hours: float = 2.
    radius: float = 200.
    max_iter: int = 20
    lat_col: str = 'latitude'
    lon_col: str = 'longitude'
    stop_col: str = 'stop_duration_h'
    name: str = 'geo_mask'
    lat_lon: tuple = None

    def applyfunc(self, row, center: tuple):
        if row[self.stop_col] < self.dwell_hours:
            return False
        d = d = distance.distance(
            center,
            (row[self.lat_col], row[self.lon_col])
        ).meters
        return bool(d <= self.radius)
    
    def genmask(self, ddf: dask.dataframe.DataFrame):

        center = None
        if self.lat_lon:
            center = self.lat_lon
        else:
            # If lat_long is empty, do some ML
            model = KMeans(
                n_clusters=1,
                init_max_iter=self.max_iter
            )
            model.fit(
                ddf[[self.lat_col, self.lon_col]].to_dask_array(lengths=True)
            )

            center = tuple(model.cluster_centers_[0])

        return ddf.map_partitions(
            lambda df: df.apply(
                self.applyfunc,
                axis=1,
                center=center
            ).rename(self.name),
            meta=(self.name, 'bool')
        )

_TMIN = datetime.time(0,0)
_TMAX = datetime.time(23,59)

class HomeHoursPremaskRule(PremaskRule):
    begin: str = '20:00'
    end: str = '07:00'
    time_col: str = 'trip_end'
    name: str = 'home_hours'

    def applyfunc(self, row, begin, end):
        t = row[self.time_col].time()
        if begin > end:
            return (_TMIN <= t < end) or (begin <= t < _TMAX)
        return begin <= t < end
    
    def genmask(self, ddf: dask.dataframe.DataFrame):
        b = pd.Timestamp(self.begin).time()
        e = pd.Timestamp(self.end).time()

        return ddf.map_partitions(
            lambda df: df.apply(
                self.applyfunc,
                axis=1,
                begin=b,
                end=e
            ).rename(self.name),
            meta=(self.name, 'bool')
        )

class TimeThresholdPremaskRule(PremaskRule):
    thresh: float = 4.
    stop_col: str = 'stop_duration_h'
    name: str = 'time_threshold'

    def genmask(self, ddf: dask.dataframe.DataFrame):

        return ddf.map_partitions(
            lambda df: df.apply(
                lambda x: x[self.stop_col] >= self.thresh,
                axis=1,
            ).rename(self.name),
            meta=(self.name, 'bool')
        ) 

class MaskPreprocesor(object):

    __slots__ = (
        '_rules'
    )

    def __init__(self, rules = []):
        self._rules = []
        if isinstance(rules, (tuple, list, set)):
            if all(isinstance(x, PremaskRule) for x in r):
                self._rules.extend(rules)
            else:
                TypeError('Members of iterable need to be of type PremaskRule.')
        elif isinstance(rules, PremaskRule):
            self._rules.append(rules)
        else:
            TypeError('rules must be eith a single or iterable of PremaskRule.')
    
    def add_rules(self, r):
        if isinstance(r, (tuple, list, set)):
            if all(isinstance(x, PremaskRule) for x in r):
                self._rules.extend(r)
            else:
                TypeError('Members of iterable need to be of type PremaskRule.')
        elif isinstance(r, PremaskRule):
            self._rules.append(r)
        else:
            TypeError('r must be eith a single or iterable of PremaskRule.')
    
    def process(self, ddf: dask.dataframe.DataFrame):

        masks = [PremaskRule().genmask(ddf)]
        r: PremaskRule
        for r in self._rules:
            masks.append(r.genmask(ddf))
        
        m0: dask.dataframe.Series
        m0 = masks[0]
        mdf = m0.to_frame()

        m: dask.dataframe.Series
        for m in masks[1:]:
            mdf[m.name] = m
        
        return mdf.map_partitions(
            lambda df: df.apply(
                lambda x: any(x),
                axis=1,
            ),
            meta=('preprocess_mask', 'bool')
        ) 
    
    @classmethod
    def create_mask(cls, ddf: dask.dataframe.DataFrame, rules: list):

        return MaskPreprocesor(rules=rules).process(ddf)

    
        



