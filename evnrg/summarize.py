import numba as nb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



@nb.njit(cache=True)
def record_energy_info(
    fleet,
    grid_ci,
    distance,
    initial_battery,
    fuel,
    battery,
    demand,
    idle_fuel,
    idle_battery,
    idle_co2,
    drive_fuel,
    drive_battery,
    drive_co2,
    stopped_battery,
    total_demand,
    total_fuel,
    total_battery,
    total_co2,
    drive_utilization):
    for i in range(distance.shape[0]):
        stopped_battery[i] = 0
        idle_battery[i] = 0
        idle_fuel[i] = 0
        drive_fuel[i] = 0
        drive_battery[i] = 0
        n_used = 0
        fuel_co2 = 0.
        total_demand[i] = 0.
        for j in range(distance.shape[1]):
            # Idle
            if distance[i, j] < 0:
                n_used += 1
                idle_fuel[i] += fuel[i, j]
                fuel_co2 += fuel[i, j] * fleet[j]['fuel_co2e']
                if i > 0:
                    idle_battery[i] += battery[i - 1, j] - battery[i, j]
                else:
                    idle_battery[i] += initial_battery[j] - battery[i, j]
                
            # Stopped
            elif distance[i, j] == 0.:
                stopped_battery[i] += battery[i, j]
                total_demand[i] += demand[i, j]
            
            # Driving
            elif distance[i, j] > 0:
                n_used += 1
                drive_fuel[i] += fuel[i, j]
                fuel_co2 += fuel[i, j] * fleet[j]['fuel_co2e']
                if i > 0:
                    drive_battery[i] += battery[i - 1, j] - battery[i, j]
                else:
                    drive_battery[i] += initial_battery[j] - battery[i, j]

        idle_co2[i] = (idle_battery[i] * grid_ci) + fuel_co2
        drive_co2[i] = (drive_battery[i] * grid_ci) + fuel_co2
        total_co2[i] = idle_co2[i] + drive_co2[i]        
        drive_utilization[i] = n_used / distance.shape[1]
        total_fuel[i] = idle_fuel[i] + drive_fuel[i]
        total_battery[i] = idle_battery[i] + drive_battery[i]

gwp_ = np.dtype([
    ('gwp_total', np.float32),
    ('gwp_fuel', np.float32),
    ('gwp_elec', np.float32)
])

def energy_info(
    fleet_name: str,
    scenario_name: str,
    fleet: np.array,
    trips: pd.DataFrame,
    grid_ci: float,
    interval_len: float,
    fuel: np.array,
    battery: np.array,
    demand: np.array):

    df = pd.DataFrame(
        data = 0.,
        columns = [
            'idle_fuel_gal',
            'idle_battery_kWh',
            'idle_ghg_kgCO2',
            'drive_fuel_gal',
            'drive_battery_kWh',
            'drive_ghg_kgCO2',
            'stopped_battery_capacity_kWh',
            'home_demand_kW',
            'total_fuel_gal',
            'total_battery_use_kWh',
            'total_ghg_kgCO2',
            'fleet_utilization'
        ],
        index = trips.index
    )

    record_energy_info(
        fleet,
        grid_ci,
        trips.values,
        fleet[:]['ev_max_batt'],
        fuel,
        battery,
        demand,
        df['idle_fuel_gal'].values,
        df['idle_battery_kWh'].values,
        df['idle_ghg_kgCO2'].values,
        df['drive_fuel_gal'].values,
        df['drive_battery_kWh'].values,
        df['drive_ghg_kgCO2'].values,
        df['stopped_battery_capacity_kWh'].values,
        df['home_demand_kW'].values,
        df['total_fuel_gal'].values,
        df['total_battery_use_kWh'].values,
        df['total_ghg_kgCO2'].values,
        df['fleet_utilization'].values
    )

    df['realtime_elec_ghg_kgCO2'] = df['home_demand_kW'] * grid_ci * (interval_len/60.)
    df['fleet'] = fleet_name
    df['scenario'] = scenario_name

    return df

def summarize_energy_info(
    df: pd.DataFrame,
    fname: str,
    scname: str):

    return pd.Series(
        data = {
            'fleet': fname,
            'scenario': scname,
            'idle_fuel_gal': df['idle_fuel_gal'].sum(),
            'idle_battery_kWh': df['idle_battery_kWh'].sum(),
            'idle_ghg_kgCO2': df['idle_ghg_kgCO2'].sum(),
            'drive_fuel_gal': df['drive_fuel_gal'].sum(),
            'drive_battery_kWh': df['drive_battery_kWh'].sum(),
            'drive_ghg_kgCO2': df['drive_ghg_kgCO2'].sum(),
            'total_fuel_gal': df['total_fuel_gal'].sum(),
            'total_battery_use_kWh': df['total_battery_use_kWh'].sum(),
            'total_ghg_kgCO2': df['total_ghg_kgCO2'].sum(),
            'total_elec_ghg_kgCO2': df['realtime_elec_ghg_kgCO2'].sum(),
            'fleet_utilization_mean': df['fleet_utilization'].mean(),
            'fleet_utilization_max': df['fleet_utilization'].max(),
        }
    )

def agg_cols(df: pd.DataFrame, fid, sid, ops: dict):

    data = {
        'fleet': fid,
        'scenario': sid
    }

    for k, oplist in ops.items():
        for v in oplist:
            if v == 'sum':
                data['total_{}'.format(k)] = df[k].sum()
            elif v == 'mean':
                data['mean_{}'.format(k)] = df[k].mean()
            elif v == 'max':
                data['max_{}'.format(k)] = df[k].max()
            elif v == 'min':
                data['min_{}'.format(k)] = df[k].min()
            elif v == 'std':
                data['std_{}'.format(k)] = df[k].std()
            elif v == 'median':
                data['median_{}'.format(k)] = df[k].median()
            elif v == 'mode':
                data['mode_{}'.format(k)] = df[k].mode()
    
    return data


def summarize_summary(df: pd.DataFrame, fid, sid):
    return {
        'mean_evse_utilization': df['evse_utilization'].mean(),
        'mean_evse_occupancy': df['evse_occupancy'].mean(),
        'mean_queue_length': df['queue_length'].mean(),
        'idle_batt_used': df['idle_batt_used'].sum(),
        'idle_fuel_used': df['idle_fuel_used'].sum(),
        'drive_batt_used': df['drive_batt_used'].sum(),
        'drive_fuel_used': df['drive_fuel_used'].sum(),
        'idle_fuel_gwp': df['idle_fuel_gwp'].sum(),
        'drive_fuel_gwp': df['drive_fuel_gwp'].sum(),
        'fleet': fid,
        'scenario': sid
    }

def apply_lambda(
    df: pd.DataFrame, 
    input_col: str,
    output_col: str,
    f: callable):

    df[output_col] = df[input_col].apply(f)

    return df

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
    df['hour'] = df.index.hour
    df['time_of_day'] = (df.index.hour.values * 100) + df.index.minute.values
    df['date'] = df.index.date
    df['weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: (x.weekday() >= 5) or (x.date() in holidays)
    )
    df['day_of_week'] = df.index.day_name()
    
    return df

def calc_gwp(df: pd.DataFrame, grid_ci: float):
    df['idle_batt_gwp'] = df['idle_batt_used'] * grid_ci
    df['drive_batt_gwp'] = df['drive_batt_used'] * grid_ci
    df['total_ghg_kgCO2'] = df['idle_fuel_gwp'] + df['drive_fuel_gwp'] + df['idle_batt_gwp'] + df['drive_batt_gwp']
    df['idle_ghg_kgCO2'] = df['idle_fuel_gwp'] + df['idle_batt_gwp']
    df['drive_ghg_kgCO2'] = df['drive_fuel_gwp'] + df['drive_batt_gwp']
    return df
        

def calc_energy_cost(
    fuel_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    fid: str,
    sid: str,
    fuel_price: float,
    elec_price: float):
    total_fuel = fuel_df.values.sum()
    total_elec = energy_df.values.sum()
    return {
        'fleet': fid,
        'scenario': sid,
        'fuel_price': fuel_price,
        'elec_price': elec_price,
        'fuel_cost': fuel_price * total_fuel,
        'elec_cost': elec_price * total_elec,
        'total_cost': (fuel_price * total_fuel) + (elec_price * total_elec)
    }

def sum_cols(df: pd.DataFrame, sname: str):

    out = pd.DataFrame(
        index = df.index,
        data = df.values.sum(axis=1),
        columns = [sname]
    )

    return out

def get_col(df: pd.DataFrame, new_name: str, colname: str):

    out = df[[colname]]

    out.columns = [new_name]

    return out
