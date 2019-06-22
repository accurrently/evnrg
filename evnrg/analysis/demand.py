import pandas as pd
import numba as nb
import numpy as np
import dask
import seaborn as sns

from .time import add_datetime_cols
from ..data import StorageInfo, Storage

def dj_prepare_demand(
    summary_dfd,
    demand_data: list,
    meta: dict,
    col: str = 'evse_demand'):
    
    df = dask.delayed(lambda x,y: x[[y]])(summary_dfd, col)
    df = dask.delayed(df.assign)(**meta)
    df = dask.delayed(add_datetime_cols)(df)
    df = dask.delayed(df.reset_index)(drop=True)

    demand_data.append(df)

def plot_demand(df: pd.DataFrame):

    sns.set(style='ticks')
    sns.plt.xlim(df['time_of_day'].min(), df['time_of_day'].max())

    g = sns.relplot(
        x='time_of_day',
        y='evse_demand',
        hue='fleet',
        kind='line',
        data=df
    )

    g.set_xticklabels(
        np.sort(df['time_of_day'].unique()),
        rotation=30,
        ha="center"
    )

    g.set_xlabels('Time of Day')
    g.set_ylabels('Demand (kW)')

    return g


def dj_save_scenario_demand(
    demand_data: list,
    result_list: list,
    scenario_id: str,
    run_id: str,
    si: StorageInfo,
    base: str = 'results',
    data_fmts: list = ['parquet', 'csv', 'records.json'],
    plot_fmt: str='svg',
    plot_dpi: str='dpi'):


    df = dask.delayed(pd.concat)(
        demand_data,
        axis=0,
        ignore_index=True
    )

    fig = dask.delayed(plot_demand)(df)

        
    for data_fmt in data_fmts:

        result_list.append(
            dask.delayed(Storage.upload_df)(
                si=si,
                df=df,
                obj_path=Storage.gen_remotepath(
                    base=base,
                    resource_type='data',
                    run_id=run_id,
                    scenario_id=scenario_id,
                    name='demand',
                    ext=data_fmt
                ), 
                fmt=data_fmt
            )
        )

    result_list.append(
        dask.delayed(Storage.upload_fig)(
            si=si,
            fig=fig,
            obj_path=Storage.gen_remotepath(
                base=base,
                resource_type='plot',
                run_id=run_id,
                scenario_id=scenario_id,
                name='demand',
                ext=fmt
            ),
            fmt=plot_fmt,
            dpi=plot_dpi
        )
    )

    return result_list


    


