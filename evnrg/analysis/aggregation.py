import pandas as pd
import numba as nb
import numpy as np
import dask

def agg_cols(
    df: pd.DataFrame,
    func: callable,
    fleet_id: str,
    scenario_id: str,
    cols: list = [],
    xcols: list = [],
    prefix: str = None):

    d = df
    if cols:
        d = d[cols]
    
    if xcols:
        d = d.drop(xcols, axis=1)
    
    
    s = func(df, axis=0)

    if prefix:
        s.index = str(prefix) + s.index.values 

    s['fleet'] = fleet_id,
    s['scenario'] = scenario_id

    return s

def dj_summary_aggregations(
    summary_dfd,
    summary_totals: list,
    summary_means: list,
    fleet_id: str,
    scenario_id: str):

    summary_totals.append(
        dask.delayed(agg_cols)(
            df=summary_dfd,
            func=pd.DataFrame.sum,
            fleet_id=fleet_id,
            scenario_id=scenario_id,
            prefix='total_'
        )
    )

    summary_means.append(
        dask.delayed(agg_cols)(
            df=summary_dfd,
            func=pd.DataFrame.mean,
            fleet_id=fleet_id,
            scenario_id=scenario_id,
            prefix='mean_'
        )
    )

    return summary_totals, summary_means