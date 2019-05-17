import random
import math
from typing import NamedTuple
import uuid

from timeit import default_timer as timer
from datetime import datetime

import numba as nb
import numpy as np
import pandas as pd


from .scenario import Scenario
from .evse import EVSEType
from .powertrain import Powertrain, PType
from .plug import DCPlug
from .types import vehicle_, evse_, make_evse_banks, make_fleet
from .utils import isin_
from .data import fleet_from_df, banks_from_df, make_mask, write_data, load_data
from .datastorage import DataHandler, DatasetInfo, StorageInfo

DCPLUG_CHADEMO = int(DCPlug.CHADEMO)
DCPLUG_TESLA = int(DCPlug.TESLA)
DCPLUG_COMBO = int(DCPlug.COMBO)
DCPLUG_NONE = int(DCPlug.NONE)


@nb.njit(cache=True)
def bank_enqueue(idx, vid, soc, fleet, queue, queue_soc):

    if fleet[vid]['ev_eff'] > 0:
        if soc < 1:
            if queue_soc:
                queue[vid] = soc
            else:
                queue[vid] = idx
    
    return queue

@nb.njit(cache=True)
def pop_low_score(queue):
    idx = -1
    for i in range(queue.shape[0]):
        x = queue[i]
        if not np.isnan(x):
            if idx == -1:
                idx = i
            elif x < queue[idx]:
                idx = i
    if (not (np.isnan(idx))) and (idx >= 0):
        queue[nb.int64(idx)] = np.nan
    return idx

@nb.njit(cache=True)
def bank_dequeue(queue, idx):
    if idx > queue.shape[0]:
        raise IndexError
    queue[idx] = np.nan
    return queue

@nb.njit(cache=True)
def dequeue_departing(distance, queue, fleet):
    for vid in range(fleet.shape[0]):
        if not (distance[vid] == 0):
            queue[vid] = np.nan
    


@nb.njit(cache=True)
def get_soc(vid, fleet, battery_nrg):
    out = 0.
    if (fleet[vid]['ev_max_batt'] > 0.):
        out = battery_nrg / fleet[vid]['ev_max_batt']
    return out



@nb.njit(cache=True)
def get_connect_evse_id(vid, soc, fleet, bank, away_bank = False):

    connected = -1

    for i in range(bank.shape[0]):
        avail = True
        if not away_bank:
            avail = not (isin_(i, fleet[:]['home_evse_id']))

        if avail and (random.random() <= bank[i]['probability']) and (soc < bank[i]['max_soc']):
            # DCFC
            if bank[i]['dc'] and (fleet[vid]['dc_max'] > 0):
                can_connect = False
                if (fleet[vid]['dc_plug'] == DCPlug.CHADEMO) and bank[i]['plug_chademo']:
                    can_connect = True
                elif (fleet[vid]['dc_plug'] == DCPlug.COMBO) and bank[i]['plug_combo']:
                    can_connect = True
                elif (fleet[vid]['dc_plug'] == DCPlug.TESLA) and bank[i]['plug_tesla']:
                    can_connect = True
                
                if can_connect:
                    connected = i
                    break
            # AC
            else:
                connected = i
                break
    return connected

@nb.njit(cache=True)
def connect_direct(vid, fleet, input_batt, bank, away_bank = False):
    # Only connect PEVs.
    if bank.shape[0] > 0:
        if fleet[vid]['ev_eff'] > 0:
            soc = 1.
            if (fleet[vid]['ev_max_batt'] > 0.):
                soc = input_batt[vid] / fleet[vid]['ev_max_batt']

            eid = get_connect_evse_id(vid, soc, fleet, bank, away_bank)

            if eid >= 0:
                if bank[eid]['dc']:
                    fleet[vid]['input_power'] = min(fleet[vid]['dc_max'], bank[eid]['power_max'])
                else:
                    fleet[vid]['input_power'] = min(fleet[vid]['ac_max'], bank[eid]['power_max'])
                fleet[vid]['input_max_soc'] = bank[eid]['max_soc']
                if away_bank:
                    fleet[vid]['away_evse_id'] = eid
                else:
                    fleet[vid]['home_evse_id'] = eid
                    bank[eid]['power'] = fleet[vid]['input_power']

@nb.njit(cache=True)
def connect_from_queue(queue, fleet, battery_state, bank):
    if bank.shape[0] > 0:
        num_connected = 0
        for vid in range(fleet.shape[0]):
            if fleet[vid]['home_evse_id'] >= 0:
                num_connected += 1
        

        failed = np.full(fleet.shape[0], -1, dtype=np.int64)
        n_failed = 0
        in_queue = 1
        while (num_connected < bank.shape[0]) and (in_queue > 0):
            low_score  = 100000
            pop_vid = -1
            in_queue = 0
            for i in range(queue.shape[0]):
                if not np.isnan(queue[i]):
                    in_queue += 1
                    if isin_(i, failed):
                        in_queue -= 1
                    elif queue[i] < low_score:
                        pop_vid = i
                        low_score = queue[i]
            
            if pop_vid >= 0:
                if (fleet[pop_vid]['ev_max_batt'] > 0.):
                    soc = battery_state[pop_vid] / fleet[pop_vid]['ev_max_batt']
                    eid = get_connect_evse_id(pop_vid, soc, fleet, bank, False)
                    if eid >= 0:
                        soc = 1.
                        if bank[eid]['dc']:
                            fleet[pop_vid]['input_power'] = min(fleet[pop_vid]['dc_max'], bank[eid]['power_max'])
                        else:
                            fleet[pop_vid]['input_power'] = min(fleet[pop_vid]['ac_max'], bank[eid]['power_max'])
                        fleet[pop_vid]['home_evse_id'] = eid
                        bank[eid]['power'] = fleet[pop_vid]['input_power']
                        fleet[pop_vid]['input_max_soc'] = bank[eid]['max_soc']

                        queue[pop_vid] = np.nan
                        num_connected += 1
                    else:
                        failed[n_failed] = pop_vid
                        n_failed += 1
                else:
                    failed[n_failed] = pop_vid
                    n_failed += 1


@nb.njit(cache=True)
def disconnect_departing(distance, fleet, home_bank):
    for vid in range(fleet.shape[0]):
        if not (distance[vid] == 0):
            home_eid = fleet[vid]['home_evse_id']  
            away_eid = fleet[vid]['away_evse_id']  
            if home_eid >= 0:
                if home_eid <= home_bank.shape[0]:
                    home_bank[home_eid]['power'] = 0.                
            
            if (home_eid >= 0) or (away_eid >= 0):
                fleet[vid]['home_evse_id'] = -1
                fleet[vid]['away_evse_id'] = -1
                fleet[vid]['input_power'] = 0.
                fleet[vid]['input_max_soc'] = 0.
    #return fleet, bank

@nb.njit(cache=True)
def disconnect_completed(battery_state, fleet, home_bank, away_bank):

    for vid in range(fleet.shape[0]):
        nrg = battery_state[vid]
        max_ = fleet[vid]['ev_max_batt']
        soc = 0.
        if max_ > 0:
            soc = nrg / max_ # get_soc(vid, fleet, nrg)
        else:
            soc = 1
        
        away_eid = fleet[vid]['away_evse_id']
        home_eid = fleet[vid]['home_evse_id']

        if (home_eid >= 0) and (soc >= home_bank[home_eid]['max_soc']):
            if home_eid < home_bank.shape[0]:
                home_bank[home_eid]['power'] = 0.
                fleet[vid]['home_evse_id'] = -1
                fleet[vid]['input_power'] = 0.
                fleet[vid]['input_max_soc'] = 0.
        if (away_eid >= 0) and (soc >= away_bank[away_eid]['max_soc']):
            if away_eid < away_bank.shape[0]:
                fleet[vid]['away_evse_id'] = -1
                fleet[vid]['input_power'] = 0.
                fleet[vid]['input_max_soc'] = 0.

    #return fleet, home_bank, away_bank

@nb.njit
def charge(current_batt, max_batt, power, max_soc, min_per_interval):
    potential = (min_per_interval / 60.0) * power
    max_nrg = max_batt * max_soc
    return min(current_batt+potential, max_nrg)


@nb.njit(cache=True)
def charge_connected(idx, battery_state, fleet, min_per_interval):

    
    for i in range(fleet.shape[0]):
        # Only charge connected
        if (fleet[i]['input_power'] > 0):
            prev_batt = fleet[i]['ev_max_batt']
            if idx > 0:
                prev_batt = battery_state[idx - 1, i]
            max_nrg = fleet[i]['ev_max_batt'] * fleet[i]['input_max_soc']
            new_nrg = prev_batt + ((float(min_per_interval) / 60.0) * fleet[i]['input_power'])
            battery_state[idx, i] = min(new_nrg, max_nrg)


@nb.njit(cache=True)
def find_next_stop(distance, idx, vid):
    stop = distance.shape[0]
    for i in range(idx, distance.shape[0]):
        if distance[i, vid] == 0.:
            stop = i
            break
    return stop

@nb.njit(cache=True)
def find_stop_end(distance, idx, vid):
    stop = distance.shape[0]
    for i in range(idx, distance.shape[0]):
        if not (distance[i, vid] == 0.):
            stop = i
            break
    return stop


@nb.njit(cache=True)
def stop_length_min(distance, min_per_interval):
    i = 0
    while (i < distance.shape[0]) and (distance[i] == 0.):
        i += 1
    return i * min_per_interval

@nb.njit(cache=True)
def defer_next_trip(distance, idx, vid, deferred):
    # Find end of stop, if necessary
    i = idx
    end = distance.shape[0]
    while (i < end) and (distance[i, vid] == 0.):
        i += 1
    while (i < end) and (not (distance[i, vid] == 0.)):
        deferred[i, vid] = distance[i, vid]
        distance[i, vid] = 0.
        i += 1

@nb.njit(cache=True)
def defer_to_target(distance, deferred):
    for i in range(distance.shape[0]):
        if distance[i] >= 0:
            deferred[i] += distance[i]
    distance[:] = 0.

@nb.njit(cache=True)
def queue_size(queue):
    x = 0
    for i in range(queue.shape[0]):
        if not (np.isnan(queue[x])):
            x += 1
    return x




@nb.njit(cache=True)
def battery_req(distance, idle_load_kw, min_per_interval, eff):
    nrg = 0.
    for idx in range(distance.shape[0]):
        # driving
        if distance[idx] > 0:
            if eff > 0:
                nrg += distance[idx] / eff
        # idle
        elif distance[idx] < 0:
            nrg += idle_load_kw * (min_per_interval / 60.0)
        idx += 1
    return nrg

@nb.njit(cache=True)
def try_defer_trips(
    vic,
    distance,
    deferred,
    mask,
    battery_nrg,
    min_per_interval,
    idle_load_kw,
    away_thresh,
    home_thresh,
    soc_buffer):

    # Only bother with BEVs
    if (vic['ev_eff'] > 0) and (vic['ice_eff'] <= 0):
        req = battery_nrg + 100
        nrg = battery_nrg * (1. - soc_buffer)
        end = distance.shape[0]
        stop = end
        i = 0
        while (i < end) and (nrg < req):

            # Find the next stop
            k = i
            while not (distance[k] == 0) and k < end:
                k += 1
            stop = k

            stop_mins = stop_length_min(distance[i:stop], min_per_interval)

            # We have an eligible stop!
            if (stop_mins >= away_thresh) or (stop_mins >= home_thresh) or mask[i]:
                req = battery_req(distance[:stop], idle_load_kw, min_per_interval, vic['ev_eff'])

                # If we have enough energy, break off
                if nrg >= req:
                    break
                
                # Defer
                j = 0
                while j < stop:
                    if distance[j] > 0:
                        deferred[j] += distance[j]
                    distance[j] = 0
                    j += 1
            
            # Move i to the next trip
            i = stop
            while distance[i] == 0.:
                i += 1
        
        #return distance, deferred


@nb.njit(cache=True)
def drive(
    distance,
    batt_state,
    battery,
    fuel,
    fleet,
    idle_load_kw,
    min_per_interval,
    drive_batt_used,
    drive_fuel_used,
    idle_batt_used,
    idle_fuel_used,
    idle_fuel_gwp,
    drive_fuel_gwp):


    for vid in range(fleet.shape[0]):

        ev_eff = fleet[vid]['ev_eff']
        ice_eff = fleet[vid]['ice_eff']
        ice_g_kwh = fleet[vid]['ice_gal_kwh']
        fuel_gwp = fleet[vid]['fuel_co2e']
        batt = batt_state[vid]

        batt_used = 0.
        fuel_used = 0.

        # driving
        if distance[vid] > 0.:
            d = distance[vid]
            
            # Handle EV
            if ev_eff > 0:
                nrg_req = d / ev_eff
                batt_used = min(batt, nrg_req)
                drive_batt_used += batt_used
                d = d - max((batt_used * ev_eff), d)
            
            # Handle ICE                
            if  ice_eff > 0.:
                fuel_used = d / ice_eff
                drive_fuel_used += fuel_used
                drive_fuel_gwp += fuel_used * fuel_gwp
        # idling
        elif distance[vid] < 0.:
            e = idle_load_kw * (min_per_interval / 60.)
            # Handle EV
            if ev_eff > 0.:
                batt_used = min(batt, e)
                e = e - max(e, batt_used)
                idle_batt_used += batt_used
            # Handle ICE
            if  ice_g_kwh > 0.:
                fuel_used = e * ice_g_kwh
                idle_fuel_used += fuel_used
                idle_fuel_gwp += fuel_used * fuel_gwp
        
        # Only set values if we actually drove        
        battery[vid] = batt - batt_used    
        fuel[vid] = fuel_used
    #return out

@nb.njit(cache=True)
def num_occupied_evse(fleet):
    n = 0
    for vid in range(fleet.shape[0]):
        eid = fleet[vid]['home_evse_id']
        if not (np.isnan(eid)):
            n += 1
    return n
        

@nb.njit(cache=True)
def evse_usage(fleet, bank, min_per_interval):

    max_power = bank[0]['bank_power_max']
    num_evse = bank.shape[0]

    out = np.zeros((3, num_evse), dtype=np.float32)
    dem = 0 
    nrg = 1 
    occ = 2
    
    for vid in range(fleet.shape[0]):
        eid = fleet[vid]['home_evse_id']
        if eid >= 0:
            out[dem, eid] = bank[eid]['power']
            out[occ, eid] = 1
            out[nrg, eid] = bank[eid]['power'] * (min_per_interval / 60.)
    
    return out

#@nb.njit(cache=True)
def simulation_loop_delayed(
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
                        if stop_time >= home_thresh_min or home_mask[idx]:
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
            distance[idx],
            bs,
            battery_state[idx,:],
            fuel_use[idx,:],
            fleet,
            idle_load_kw,
            interval_min,
            drive_batt_used[idx],
            drive_fuel_used[idx],
            idle_batt_used[idx],
            idle_fuel_used[idx],
            idle_fuel_gwp[idx],
            drive_fuel_gwp[idx])
        
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

    
    return (
        # Fuel
        pd.DataFrame(
            data=fuel_use,
            columns=trips.columns,
            index=trips.index
        ),
        # Battery state
        pd.DataFrame(
            data=battery_state,
            columns=trips.columns,
            index=trips.index
        ),
        # Deferred
        pd.DataFrame(
            data=deferred,
            columns=trips.columns,
            index=trips.index
        ),
        # Elec Demand
        pd.DataFrame(
            data=elec_demand,
            columns=np.array(home_bank_names),
            index=trips.index
        ),
        # Elec Energy
        pd.DataFrame(
            data=elec_energy,
            columns=np.array(home_bank_names),
            index=trips.index
        ),

        # Summary
        pd.DataFrame(
            data={
                'evse_utilization': utilization,
                'evse_occupancy': occupancy,
                'queue_length': queue_length,
                'idle_batt_used': idle_batt_used,
                'idle_fuel_used': idle_fuel_used,
                'drive_batt_used': drive_batt_used,
                'drive_fuel_used': drive_fuel_used,
                'idle_fuel_gwp': idle_fuel_gwp
                'drive_fuel_gwp': drive_fuel_gwp
            },
            index=trips.index
        ),
        # Arrays and DFS
        fleet,
        home_bank,
        trips
    )

class SimResult(NamedTuple):
    fuel: pd.DataFrame
    battery: pd.DataFrame
    deferred: pd.DataFrame
    demand: pd.DataFrame
    energy: pd.DataFrame
    evse_info: pd.DataFrame
    fleet: np.array
    home_bank: np.array
    trips: pd.DataFrame

def run_simulation(ds: DatasetInfo, sc: Scenario, storage_info: StorageInfo):
    """Runs a simulation with a given scenario by
    downloading data, running the simulation, and uploading results.

    Args:
        sc (Scenario): The scenario structure to use.
    
    Returns:
        A `SimulationResult` with all the relevant energy data.
    """

    sim_start = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    st = DataHandler(storage_info)

    try:
        # First pull down the data
        
        df = st.read_data(ds.obj_path)

        df_index = df.index

        rows = len(df_index)

        interval_len = df.index.to_series().diff().min().seconds / 60.0

        # Basic rules creation for now

        mask = pd.Series(
            np.zeros(df.values.shape[0], dtype=np.bool_),
            index=df.index
        )

        for mask_rule in sc.home_mask_rules:
            if mask_rule.get('type') == 'time':
                begin = mask_rule.get('begin', '23:55')
                end = mask_rule.get('end', '00:00')
                mask[df.between_time(begin, end).index] = True

        # Create the fleet
        fleet = fleet_from_df(df, sc.powertrains, sc.distribution)

        fleet_size = fleet.shape[0]

        home_banks = make_evse_banks(sc.home_banks, fleet_size)

        away_banks = make_evse_banks(sc.away_banks, fleet_size)
        away_banks[:]['power'] = away_banks[:]['power_max']

        num_banks = home_banks.shape[0]

        use_soc_queue = False

        timer_begin = timer()
        out =  simulation_loop_delayed(
            df,
            fleet,
            home_banks,
            away_banks,
            mask.values,
            interval_len,
            sc.home_threshold_min,
            sc.away_threshold_min,
            sc.idle_load_kw,
            use_soc_queue,
            sc.soc_deferment_buffer
        )
    except Exception as e:
        raise e
    finally:
        st.cleanup()
    return SimResult(*out)