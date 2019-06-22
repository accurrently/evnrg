import random
import math
from typing import NamedTuple
import uuid

from datetime import datetime


import numba as nb
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from .types import SimResult
from .jitfuncs import *
from ..fuels import _KWH_PER_MJ

def sim_loop(
    trips: pd.DataFrame,
    fleet: np.array,
    home_bank: np.array,
    away_bank: np.array,
    home_mask: np.array,
    interval_min: float,
    home_thresh_min: float,
    away_thresh_min: float,
    idle_load_kw: float,
    queue_soc: bool = False,
    soc_buffer: float = 0.2):

    run_begin = datetime.now()

    if (not isinstance(trips, pd.DataFrame)) or trips.empty:
        raise ValueError(
            'There\'s no data!'
        )

    distance = trips.values

    nrows = distance.shape[0]
    nvehicles = fleet.shape[0]
    dshape = (nrows, nvehicles)
    nevse = home_bank.shape[0]

    fuel_use = np.zeros(dshape, dtype=np.float32)
    battery_state = np.zeros(dshape, dtype=np.float32)

    elec_demand = np.zeros(nrows, dtype=np.float32)
    elec_energy = np.zeros(nrows, dtype=np.float32)
    if nevse > 0:
        elec_demand = np.zeros((nrows,nevse), dtype=np.float32)
        elec_energy = np.zeros((nrows, nevse), dtype=np.float32)
    occupancy = np.zeros(nrows, dtype=np.float32)
    utilization = np.zeros(nrows, dtype=np.float32)
    deferred = np.zeros(dshape, dtype=np.float32)
    queue_length = np.zeros(nrows, dtype=np.int64)
    connected_home_evse = np.full((nrows, nvehicles), -1, dtype=np.int64)
    drive_batt_used = np.zeros(nrows, dtype=np.float32)
    drive_fuel_used = np.zeros(nrows, dtype=np.float32)
    idle_batt_used = np.zeros(nrows, dtype=np.float32)
    idle_fuel_used = np.zeros(nrows, dtype=np.float32)
    idle_fuel_gwp = np.zeros(nrows, dtype=np.float32)
    drive_fuel_gwp = np.zeros(nrows, dtype=np.float32)

    pct_driving = np.zeros(nrows, dtype=np.float32)
    pct_idle = np.zeros(nrows, dtype=np.float32)
    pct_stopped = np.zeros(nrows, dtype=np.float32),
    pct_charging = np.zeros(nrows, dtype=np.float32)
    stopped_battery = np.zeros(nrows, dtype=np.float32)
    pct_batt_available = np.zeros(nrows, dtype=np.float32)

    queue = np.full(nvehicles, np.nan, dtype=np.float32)

    for idx in range(nrows):

        # Disconnect and dequeue departing vehicles
        dequeue_departing(distance[idx], queue, fleet)
        disconnect_departing(distance[idx], fleet, home_bank)
        
        # Disconnect completed vehicles
        if idx > 0:
            disconnect_completed(battery_state[idx-1], fleet, home_bank, away_bank)
        
        # Do drive and stop stuff
        if idx > 0:

            for vid in range(nvehicles):

                curr_d = distance[idx, vid]
                prev_d = distance[idx -1, vid]

                # Stop stufffor PEVs
                if (curr_d == 0.) and (fleet[vid]['ev_eff'] > 0):

                    # New stop
                    if not ( prev_d == 0.):
                        stop_time = stop_length_min(distance[idx:,vid], interval_min)
                        soc = get_soc(vid, fleet, battery_state[idx - 1, vid])
                        
                        # Home stop
                        if stop_time >= home_thresh_min or home_mask[idx][vid]:
                            # Test to make sure we even have a home bank!
                            if (home_bank.shape[0] > 0) :
                                queue = bank_enqueue(idx, vid, soc, fleet, queue, queue_soc)
                            
                        
                        # Away stop
                        elif stop_time >= away_thresh_min:
                            connect_direct(vid, fleet, distance[idx-1], away_bank, away_bank = True)
                    
                    # About to depart
                    elif idx < nrows - 1:
                        # ^^ Required check to make sure we don't overrun the array.
                        # Remember that Numba will happily overrun without question.
                        if (not (distance[idx + 1, vid] == 0.)) and (fleet[vid]['ice_eff'] <= 0.):
                            # ^^ Double-check that this is only for BEVs
                            try_defer_trips(
                                fleet[vid],
                                distance[idx+1:, vid],
                                deferred[idx+1:, vid],
                                home_mask[idx+1:],
                                battery_state[idx - 1, vid],
                                interval_min,
                                idle_load_kw,
                                away_thresh_min,
                                home_thresh_min,
                                soc_buffer
                            )
                            #distance[idx+1:, vid] = dist_a
                            #deferred[idx+1, vid] = defer_a

        bs = fleet[:]['ev_max_batt']
        if idx > 0:
            bs = battery_state[idx - 1,:]

        # Drive
        drive(
            idx,
            distance[idx],
            bs,
            battery_state[idx,:],
            fuel_use[idx,:],
            fleet,
            idle_load_kw,
            interval_min,
            drive_batt_used,
            drive_fuel_used,
            idle_batt_used,
            idle_fuel_used,
            idle_fuel_gwp,
            drive_fuel_gwp)
        
        # Process queue
        connect_from_queue(queue, fleet, bs, home_bank)
        queue_length[idx] = queue_size(queue)
        connected_home_evse[idx] = fleet[:]['home_evse_id']

        # Charge connected vehicles  
        charge_connected(idx, battery_state, fleet, interval_min)
        


        # Record usage
        if nevse > 0:
            usage_info = evse_usage(fleet, home_bank, interval_min)
            elec_demand[idx,:] = usage_info[0,:]
            elec_energy[idx,:] = usage_info[1,:]
            
            occupancy[idx] = float(usage_info[2,:].sum()) / float(nevse)
        
        
        
            util = 0

            if home_bank.shape[0] > 0:
                if home_bank[0]['bank_power_max'] > 0:
                    util = usage_info[0,:].sum() / home_bank[0]['bank_power_max']
            
            utilization[idx] = util
        else:
            elec_demand[idx] = 0
            elec_energy[idx] = 0
            occupancy[idx] = 0
            utilization[idx] = 0
        
        pct_dr, pct_id, pct_st, \
        pct_ch, stp_bat, \
        pct_batt_avail = drive_usage(fleet, distance[idx,:], battery_state[idx,:])

        pct_driving[idx] = pct_dr
        pct_idle[idx] = pct_id
        pct_stopped[idx] = pct_st
        pct_charging[idx] = pct_ch
        stopped_battery[idx] = stp_bat
        pct_batt_available[idx] = pct_batt_avail

    
    home_bank_names = []
    if home_bank.shape[0] > 0:
        for i in range(home_bank.shape[0]):
            if home_bank[i]['dc']:
                home_bank_names.append(
                    'evse{}_{}kW_dc'.format(
                        i,
                        home_bank[i]['power_max']
                    )
                )
            else:
                home_bank_names.append(
                    'evse{}_{}kW_ac'.format(
                        i,
                        home_bank[i]['power_max']
                    )
                )
    else:
        home_bank_names = ['evse_null']
    
    if np.ndim(elec_demand) < 2:
        elec_demand = np.zeros((nrows,1), dtype=np.float32)
    
    if np.ndim(elec_energy) < 2:
        elec_energy = np.zeros((nrows,1), dtype=np.float32)
    
    run_end = datetime.now()

    tx: pd.DatetimeIndex
    tx = trips.index
    cal = calendar()
    holidays = cal.holidays(start=tx.date.min(), end=tx.date.max())

    r = SimResult(
        # Fuel
        fuel=pd.DataFrame(
            data=fuel_use,
            columns=trips.columns,
            index=trips.index
        ),
        # Battery state
        battery=pd.DataFrame(
            data=battery_state,
            columns=trips.columns,
            index=trips.index
        ),
        # Deferred
        deferred=pd.DataFrame(
            data=deferred,
            columns=trips.columns,
            index=trips.index
        ),
        # Elec Demand
        demand=pd.DataFrame(
            data=elec_demand,
            columns=np.array(home_bank_names),
            index=trips.index
        ),
        # Elec Energy
        energy=pd.DataFrame(
            data=elec_energy,
            columns=np.array(home_bank_names),
            index=trips.index
        ),
        # Summary
        summary=pd.DataFrame(
            data={

                # Time stuff, for indexing
                'time': tx.time,
                'hour': tx.hour.values,
                'minute': tx.minute.values,
                'hour_float': tx.hour.values + (tx.minute.values/60.),
                'time_mil': tx.hour.values * 100) + tx.minute.values,
                'date': tx.date,
                'weekend_or_holiday': tx.to_series().apply(
                    lambda x: (x.weekday() >= 5) or (x.date() in holidays)
                ).values,
                'day_name': tx.day_name().values,
                'weekday': tx.weekday.values,
                
                # Idle energy and gwp
                'idle_batt_used': idle_batt_used,
                'idle_fuel_used': idle_fuel_used,
                'idle_fuel_gwp': idle_fuel_gwp,

                # Drive energy and gwp
                'drive_batt_used': drive_batt_used,
                'drive_fuel_used': drive_fuel_used,
                'drive_fuel_gwp': drive_fuel_gwp,

                # EVSE information
                'evse_demand': np.sum(elec_demand, axis=1),
                'evse_energy': np.sum(elec_energy, axis=1),
                'evse_energy_mj': np.sum(elec_energy, axis=1) / _KWH_PER_MJ,
                'evse_utilization': utilization,
                'evse_occupancy': occupancy,
                'evse_queue_length': queue_length,

                # Fleet state info
                'fleet_drive_pct': pct_driving,
                'fleet_idle_pct': pct_idle,
                'fleet_stopped_pct': pct_stopped.
                'fleet_charging_pct': pct_charging,
                'fleet_stopped_battery_capacity': stopped_battery,
                'fleet_stopped_battery_pct': pct_batt_available,
                'fleet_deferred_distance': np.sum(deferred, axis=1),
                'fleet_battery_state': np.sum(battery_state, axis=1),
                'fleet_fuel_use': np.sum(fuel_use, axis=1),
                'fleet_fuel_use_mj': np.sum(fuel_use * fleet[:]['fuel_mj_gal'], axis=1)
            },
            index=tx
        ),
        # Arrays and DFS
        fleet=fleet,
        home_bank=home_bank,
        trips=trips,

        # Other data
        run_begin = run_begin,
        run_duration = (run_end - run_begin).total_seconds()
    )
    return r