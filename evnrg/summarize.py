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
                idle_fuel += fuel[i, j]
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
 #   """
 #   ,
 #   ,
 #   distance,
 #   initial_battery,
 #   fuel,
 #   battery,
 #   demand,
 #   idle_fuel,
 #   idle_battery,
 #   idle_co2,
 #   drive_fuel,
 #   drive_battery,
 #   drive_co2,
 #   stopped_battery,
 #   total_demand,
 #   total_fuel,
 #   total_battery,
 #   total_co2,
 #   drive_utilization
 #   """

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


def add_datetime_cols( df: pd.DataFrame ):
    cal = calendar()
    holidays = cal.holidays(start=df.index.date.min(), end=df.index.date.max())

    df['time'] = df.index.time
    df['hour'] = df.index.hour.values + (df.index.minute.values / 60.)
    df['hour24'] = (df.index.hour.values * 100) + df.index.minute.values
    df['date'] = df.index.date
    df['weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: (x.weekday() >= 5) or (x.date() in holidays)
    )
    df['day_of_week'] = df.index.day_name()
    
    return df


def energy_pricing(
    s: pd.Series,
    e_prices: list,
    f_prices: list):

    frames = []

    for ep in e_prices:
        
        fr = pd.DataFrame(
            [s]*len(f_prices)
        )
        fr['elec_price'] = ep
        fr['fuel_price'] = f_prices
        frames.append(fr)

    df = pd.concat(frames, axis=0, ignore_index=True)

    df['idle_fuel_cost'] = df['idle_fuel_gal'] * df['fuel_price']
    df['idle_elec_cost'] = df['idle_battery_kWh'] * df['elec_price']
    df['drive_fuel_cost'] = df['drive_fuel_gal'] * df['fuel_price']
    df['drive_elec_cost'] = df['drive_battery_kWh'] * df['elec_price']
    df['used_elec_cost'] = df['idle_elec_cost'] + df['drive_elec_cost']
    df['used_fuel_cost'] = df['idle_fuel_cost'] + df['drive_fuel_cost']
    df['total_running_cost'] = df['used_elec_cost'] + df['used_fuel_cost']

    return df












