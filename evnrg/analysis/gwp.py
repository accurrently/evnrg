import pandas as pd
import numba as nb
import numpy as np
import dask


def calc_gwp(df: pd.DataFrame, grid_ci: float):
    df['idle_batt_gwp'] = df['idle_batt_used'] * grid_ci
    df['drive_batt_gwp'] = df['drive_batt_used'] * grid_ci
    df['total_ghg_kgCO2'] = df['idle_fuel_gwp'] + df['drive_fuel_gwp'] + df['idle_batt_gwp'] + df['drive_batt_gwp']
    df['idle_ghg_kgCO2'] = df['idle_fuel_gwp'] + df['idle_batt_gwp']
    df['drive_ghg_kgCO2'] = df['drive_fuel_gwp'] + df['drive_batt_gwp']
    return df



