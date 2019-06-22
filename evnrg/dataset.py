from typing import NamedTuple
import pandas as pd
import numpy as np
import numba as nb
import dask
import dask.dataframe as dd

from .obj.schemas import ColumnMap

class DatasetInfo(NamedTuple):
    dataset_id: str
    obj_path: str

@nb.njit
def _nb_create_dprof(
    a: np.array,
    idx: np.array,
    beg_arr: np.array,
    end_arr: np.array,
    dst_arr: np.array):

    lsz = beg_arr.shape[0]
    psz = idx.shape[0]

    for i in range(lsz):
        j = 0
        while idx[j] < beg_arr[i] and j < psz:
            j += 1
        
        b = j
        
        while idx[j] < end_arr[i] and j < psz:
            j += 1
        
        e = j
        
        n = e - b
        
        if n > 0:
            x = dst_arr[i]
            if x >= 0:
                a[b:e] = x / n
            else:
                a[b:e] = -1

@nb.njit
def _nb_create_mprof(
    a: np.array,
    idx: np.array,
    beg_arr: np.array,
    end_arr: np.array,
    msk_arr: np.array):

    lsz = beg_arr.shape[0]

    psz = a.shape[0]

    for i in range(lsz):
        j = 0
        while idx[j] < end_arr[i]:
            j += 1
        
        b = j
        
        if i < lsz - 1:
            while idx[j] < beg_arr[i+1]:
                j += 1
        else:
            while j < psz:
                j += 1
        
        e = j
        
        a[b:e] = msk_arr[i]

def _make_drange_arr(tsmin, tsmax, omin, omax, freq='5min',):

    omin = 0
    omax = 0

    return pd.date_range(
        pd.Timestamp(tsmin)+ omin,
        tsmax + omax,
        freq=freq
    ).values[:-1]

def _make_empty_ddf(tsmin, tsmax, freq='5min'):
    tmin = pd.Timestamp(tsmin)+ pd.DateOffset(days=0, normalize=True) 
    tmax = tsmax + pd.DateOffset(days=1, normalize=True)
    return pd.DataFrame(
        index = pd.date_range(tmin, tmax, freq=freq).values[:-1]
    )

def _make_prof_series(
    name: str,
    df: pd.DataFrame,
    ts_begin,
    ts_end,
    freq: str = '5min'):

    beg = ts_begin + pd.DateOffset(days=0, normalize=True)
    end = ts_end + pd.DateOffset(days=1, normalize=True)
    
    return pd.Series(
        data=0.,
        index=pd.date_range(beg, end, freq=freq),
        dtype=np.float32,
        name=name
    )


def _make_dprof_series(
    df: pd.DataFrame,
    s: pd.Series,
    cm: ColumnMap):
   
    _nb_create_dprof(
        idx=s.index.values,
        a=s.values,
        beg_arr=df[cm.trip_begin].values,
        end_arr=df[cm.trip_end].values,
        dst_arr=df[cm.distance].values
    )
    
    return s

def _make_mprof_series(
    df: pd.DataFrame,
    s: pd.Series,
    cm: ColumnMap):

    _nb_create_mprof(
        a=s.values,
        idx=s.index.values,
        beg_arr=df[cm.trip_begin].values,
        end_arr=df[cm.trip_end].values,
        msk_arr=df[cm.home_mask].values
    )


class ProfileGenerator(object):

    __slots__ = (
        'cols',
        'freq',
        'offset',
        'normalize_offset',
        '_omin',
        '_omax',
        '_part_freq'
    )

    def __init__(
        self,
        cols: ColumnMap = ColumnMap(),
        freq: str = '5min',
        offset: str = 'months'):
        
        self.normalize_offset = True
        self.cols = cols
        self.offset = offset
        self._omin = 0
        self._omax = 1
        self.freq = freq
        self._part_freq = 'M'                
    
    def date_offset(self, ts: pd.Timestamp, val = 'min'):
        x = self._omin
        if isinstance(val, str):
            if val == 'min':
                x = self._omin
            elif val == 'max':
                x = self._omax
            else:
                raise ValueError('If using str, only allowed values for val are \'max\' and \'min\'.')
        elif isinstance(int):
            x = val
        else:
            raise TypeError('val must be of type str or int')
        
        ts = pd.Timestamp(ts)
        
        o = pd.DateOffset(**{
            self.offset: x,
            'normalize': self.normalize_offset
        })

        return ts + o

    def date_range(self, ts_min: pd.Timestamp, ts_max: pd.Timestamp, freq: str = None):
        if freq is None:
            freq = self.freq

        return pd.date_range(
            start=self.date_offset(ts_min, 'min'),
            end=self.date_offset(ts_max, 'max')
            freq=freq
        )

    def distance_profile(self, df, ts_min: pd.Timestamp = None, ts_max: pd.Timestamp = None):
        if ts_min is None:
            ts_min = df[self.cols.trip_begin].min()
        else:
            ts_min = pd.Timestamp(ts_min)
        if ts_max is None:
            ts_max = df[self.cols.trip_end].max()
        else:
            ts_max = pd.Timestamp(ts_max)

        delayed_df = dask.delayed(pd.DataFrame)(
            index = self.date_range(ts_min, ts_max, self.freq)
        )

        part_range = self.date_range(ts_min, ts_max, freq=self._part_freq)

        





def expand_trip(
    start: pd.Timestamp,
    end: pd.Timestamp,
    val: float,
    out_df: pd.DataFrame,
    vid: str):

    b = start.round(out_df.index.freq)
    e = end.round(out_df.index.freq)
    idx = pd.date_range(b, e, freq=out_df.index.freq)
    if len(idx) > 1:
        i = idx[:-1]
        out_df.loc[i,col] = float(val/len(i)) if val >= 0 else -1

def generate_distance_profile(df: pd.DataFrame, freq: str = '5min', cm: ColumnMap = ColumnMap()):

    # Ensure proper types
    df[cm.vehicle_id] = df[cm.vehicle_id].apply(str)
    df[cm.trip_begin] = df[cm.trip_begin].apply(pd.Timestamp)
    df[cm.trip_end] = df[cm.trip_end].apply(pd.Timestamp)
    
    min_ts = df[cm.trip_begin].min()
    ts_begin = pd.Timestamp('{}-{}-01T00:00:00'.format(min_ts.year, min_ts.month))

    max_ts = df[cm.trip_end].max()
    the_month = 1 if max_ts.month >= 12 else max_ts.month + 1
    the_year = max_ts.year + 1 if the_month <= 1 else max_ts.year
    ts_end = pd.Timestamp('{}-{}-01T00:00:00'.format(the_year, the_month))

    idx = pd.date_range(ts_begin, ts_end, freq=freq)

    out = pd.DataFrame(index=idx)

    for vid in df[cm.vehicle_id].unique():
        out[vid] = 0.
    
    df.apply(
        lambda x: expand_trip(
            x[cm.trip_begin],
            x[cm.trip_end],
            x[cm.distance],
            out,
            x[cm.vehicle_id]
        ),
        axis = 1
    )

    return out
