import pandas as pd
import numba as nb
import numpy as np
import dask

def calc_energy_cost(
    fuel_qty,
    elec_qty,
    fid: str,
    sid: str,
    fuel_price: float,
    elec_price: float):
    return {
        'fleet': fid,
        'scenario': sid,
        'fuel_price': fuel_price,
        'elec_price': elec_price,
        'fuel_cost': fuel_price * fuel_qty,
        'elec_cost': elec_price * elec_qty,
        'total_cost': (fuel_price * fuel_qty) + (elec_price * elec_qty)
    }

def energy_cost_from_df(df: pd.DataFrame, fuel_col: str, elec_col: str, fuel_price: float, elec_price: float):
    return df.apply(
        lambda x: calc_energy_cost(
            x[fuel_col],
            x[elec_col],
            x['fleet'],
            x['scenario'],
            fuel_price,
            elec_price
        ),
        axis=1,
        result_type='expand'
    )

def af_energy_costs(
    s: pd.series,
    fprice: float,
    eprice: float):

    keep = [
        'fleet',
        'scenario',
        'total_idle_batt_used',
        'total_idle_fuel_used',
        'total_drive_batt_used',
        'total_drive_fuel_used',
        'total_evse_energy',

    ]

    t = pd.Series({
        'idle_batt_cost': s['total_idle_batt_used'] * eprice,
        'idle_fuel_cost': s['total_idle_fuel_used'] * fprice,

        'drive_batt_cost': s['total_drive_batt_used'] * eprice,
        'drive_fuel_cost': s['total_drive_fuel_used'] * fprice,

        'total_batt_cost': (s['total_batt_used'] +  s['total_idle_batt_used']) * eprice,
        'total_evse_energy_cost': s['total_evse_energy'] * eprice,
        'total_fuel_cost': s['total_fleet_fuel_use'] * fprice,

    })

    return pd.concat([s[sel],t])

delayed_cost_from_df = dask.delayed(energy_cost_from_df)

def dj_process_energy_costs(delayed_sumary_df, )

