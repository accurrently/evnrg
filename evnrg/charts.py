import numpy as np
import pandas as pd
import seaborn as sns

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from .datastorage import StorageInfo, DataHandler
from .data import write_data

def chart_demand(
    demand_cols: list, 
    fleet_names: list, 
    si: StorageInfo,
    basepath: str,
    meta: dict = {}):

    dh = DataHandler(si)
    p = si.gen_temp_path('png')

    cal = calendar()
    holidays = cal.holidays(start=df.index.date.min(), end=df.index.date.max())

    df = pd.concat(demand_cols, axis=1)
    df.columns = fleet_names
    df['time'] = df.index.time
    df['weekend_or_holiday'] = (df.index.weekday >= 5) or (df.index.date in holidays)

    df = df.reset_index(drop=True)

    df = pd.melt(
        df,
        id_vars=[
            'time',
            'weekend_or_holiday'
        ],
        value_vars=fleet_names,
        var_name='fleet',
        value_name='demand_kW'
    )

    g = sns.FacetGrid(df, col="weekend_or_holiday")
    g.map(sns.lineplot, x='time', y='demand_kW', hue='fleet')
    g.add_legend()
    g.savefig(p)

    out = dh.upload_file(
        p,
        basepath + '/' + 'demand.png',
        'png',
        meta=meta
    )

    dh.cleanup()

    return out

def melt_and_plot_facets(
    df: pd.DataFrame,
    si: StorageInfo,
    id_vars: list,
    val_vars: list,
    var_name: str,
    val_name: str,
    facet_opts: dict,
    map_func: callable,
    map_opts: dict,
    basepath: str,
    name: str,
    use_legend: bool = True,
    meta: dict = {}):

    dh = DataHandler(si)
    p = si.gen_temp_path('png')

    fr = df.melt(
        id_vars=id_vars,
        value_vars=val_vars,
        var_name=var_name,
        value_name=val_name
    )

    g = sns.FacetGrid(
        fr,
        **facet_opts
    )
    
    g.map(map_func, **map_opts)
    g.add_legend()
    g.savefig(p)

    out = dh.upload_file(
        p,
        basepath + '/' + name + '.png',
        'png',
        meta=meta
    )

    dh.cleanup()

    return out




