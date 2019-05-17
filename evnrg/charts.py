import numpy as np
import pandas as pd
import seaborn as sns
import tempfile
import uuid

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

def upload_chart(
    fig,
    si: StorageInfo,
    basepath: str,
    name: str,
    ext: str = 'svg'):

    dh = DataHandler(si)

    with tempfile.TemporaryDirectory() as tdir:
        basefile = tdir + '/' + uuid.uuid4().hex
        imgfile = basefile + '.' + ext
        fig.savefig(imgfile, dpi=300)

        out = dh.upload_file(
            imgfile,
            basepath + '/' + name + '.' + ext,
            ext
        )
        return out

    return {}



def plot_demand(df: pd.DataFrame, si: StorageInfo, basepath: str):

    sns.set(style='ticks')
    g = sns.relplot(
        x='time_of_day',
        y='demand',
        hue='fleet',
        col='weekend_or_holiday',
        row='scenario',
        kind='line',
        data=df
    )

    g.set_xticklabels(rotation=30)

    return upload_chart(
        fig=g,
        si=si,
        basepath=basepath,
        name='demand'
    )

def plot_bar(df: pd.DataFrame, si: StorageInfo, 
    basepath: str, y: str, x: str, col: str, name: str, wrap: int=4):
    
    g = sns.catplot(
        col=col,
        x=x,
        y=y,
        kind='bar',
        data=df,
        col_wrap=wrap
    )

    g.set_xticklabels(rotation=30)

    return upload_chart(
        fig=g,
        si=si,
        basepath=basepath,
        name=name
    )



def plot_facets(
    df: pd.DataFrame,
    si: StorageInfo,
    facet_opts: dict,
    map_func: callable,
    map_opts: dict,
    basepath: str,
    name: str,
    use_legend: bool = True,
    meta: dict = {}):

    current_palette = sns.color_palette()
    sns.palplot(current_palette)

    g = sns.FacetGrid(
        df,
        margin_titles=True,
        **facet_opts
    )

    g.map_dataframe(map_func, **map_opts)
    g.add_legend()

    dh = DataHandler(si)

    with tempfile.TemporaryDirectory() as tdir:
        basefile = tdir + '/' + uuid.uuid4().hex
        svgfile = basefile + '.svg'
        g.savefig(svgfile, dpi=300)

        out = dh.upload_file(
            svgfile,
            basepath + '/' + name + '.svg',
            'svg',
            meta=meta
        )
        return out      

    return {}





