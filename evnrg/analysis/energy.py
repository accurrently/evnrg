import pandas as pd
import numba as nb
import numpy as np
import dask

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