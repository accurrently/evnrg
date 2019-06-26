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

def _ts_floor(ts: pd.Timestamp, freq: str = 'month'):

    d = {
        'month': lambda: pd.Timedelta('1 day') * (ts.day - 1),
        'week': lambda: pd.Timedelta('1 day') * ts.weekday,
        'day': lambda: 0
    }

    f = d.get(freq)

    if not f:
        raise ValueError(
            """freq must be month, week, or day."""
        )
    
    return ts.normalize() - f()

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

class TripLog(object):

    __slots__ = (
        'df',
        'div_freq',
        'cm',
        'freq'
    )

    def __init__(self, 
        data,
        fmt: str='parquet',
        chunksize: int=1000,
        division_freq: str='month',
        freq: str='5min',
        column_map: ColumnMap=ColumnMap()):

        self.cm = column_map
        self.div_freq = division_freq
        self.freq = freq

        if isinstance(data, str):
            if fmt == 'parquet':
                self.df = dd.read_parquet(data)
            elif fmt == 'csv':
                self.df = dd.read_csv(data)
            else:
                raise ValueError(
                    """Invalid file format. 
                    If reading from file, use 'csv' or 'parquet'.
                    """
                )
        elif isinstance(data, dd.DataFrame):
            self.df = data
        elif isinstance(data, pd.DataFrame):
            self.df = dd.from_pandas(data, chunksize=chunksize)
        elif isinstance(data, list):
            self.df = dd.from_delayed(data)
        else:
            raise TypeError(
                """Data must be of type str, 
                dask.DataFrame, 
                pandas.DataFrame, 
                or list of delayed pandas.DataFrame.
                """
            )
    
    def date_range(self, freq: str = '5min'):
        
        b = _ts_floor(
            self.df[self.cm.trip_begin].min(),
            freq = self.div_freq
        )

        e_ts = self.df[self.cm.trip_end].max() + pd.DateOffset(**{self.div_freq: 1})

        e = _ts_floor(
            e_ts,
            freq = self.div_freq
        )

        return pd.date_range(b, e, freq=freq)

    def _make_empty_df(self):
        return pd.DataFrame(
            index = self.date_range(freq=self.freq)
        )

    def _make_prof_series(self, name: str):
        
        return pd.Series(
            data=0.,
            index=self.date_range(freq=self.freq),
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
