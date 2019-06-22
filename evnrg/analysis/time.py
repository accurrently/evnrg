import pandas as pd
import numba as nb
import numpy as np
import dask
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def add_time_cols(df: pd.DataFrame):
    cal = calendar()
    holidays = cal.holidays(start=df.index.date.min(), end=df.index.date.max())
    df['time_of_day'] = (df.index.hour.values * 100) + df.index.minute.values
    df['weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: (x.weekday() >= 5) or (x.date() in holidays)
    )
    return df

def add_datetime_cols( df: pd.DataFrame ):
    cal = calendar()
    holidays = cal.holidays(start=df.index.date.min(), end=df.index.date.max())

    df['time'] = df.index.time
    df['hour'] = df.index.hour.values
    df['minute'] = df.index.minute.values
    df['hourf'] = df.index.hour.values + (df.index.minute.values/60.)
    df['time_of_day'] = (df.index.hour.values * 100) + df.index.minute.values
    df['date'] = df.index.date
    df['weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: (x.weekday() >= 5) or (x.date() in holidays)
    )
    df['day_of_week'] = df.index.day_name()
    df['day_of_week_num'] = df.index.weekday()
    
    return df


