import random
import math
from typing import NamedTuple
import uuid

import numba as nb
import numpy as np
import pandas as pd


from .scenario import Scenario
from .datastorage import DatasetInfo, StorageInfo, DataHandler
from .evse import EVSE, EVSEType
from .bank import Bank
from .powertrain import Powertrain, PType
from .plug import DCPlug



ICEV = 0
PHEV = 1
BEV = 2



EVSE_POWER_MAX = 0
EVSE_BANK_POWER_MAX = 1
EVSE_POWER = 2
EVSE_MAX_SOC = 3
EVSE_PROBABILITY = 4
EVSE_DC = 5
DCPLUG_CHADEMO = 6 # Index for EVSE
DCPLUG_COMBO = 7
DCPLUG_TESLA = 8
DCPLUG_NONE = 0 # None Type

evse_ = np.dtype([
    ('bank_id', np.uint64, 1),
    ('power_max', np.float32, 1),
    ('bank_power_max', np.float32, 1),
    ('power', np.float32, 1),
    ('max_soc', np.float32, 1),
    ('probability', np.float32, 1),
    ('dc', np.bool_, 1),
    ('plug_chademo', np.bool_, 1),
    ('plug_combo', np.bool_, 1),
    ('plug_tesla', np.bool_, 1)
])

nb_evse_ = nb.from_dtype(evse_)

def make_evse_banks(evse_banks: list, fleet_size: int):

    n_evse = 0

    bank_ids = []
    power_max = []
    bank_power_max = []
    max_soc = []
    probability = []
    dc = []
    plug_chademo = []
    plug_combo = []
    plug_tesla = []  

    bank_id = 0
    for bank in evse_banks:
        
        bank_running_power = 0
        n_in_bank = 0
        for e, min_, max_, prop_ in bank['evse']:
            e: EVSEType
            copies = int(fleet_size * prop_)
            if max_ > 0:
                copies = int(min(max_, copies))
            copies = int(max(min_, copies))

            for i in range(copies):
                bank_ids.append(bank_id),
                power_max.append(e.max_power)
                
                max_soc.append(e.max_soc)
                probability.append(bank.get('probability', 1.))
                dc.append(e.dc)
                plug_chademo.append(bool(DCPlug.CHADEMO in e.dc_plugs))
                plug_combo.append(bool(DCPlug.COMBO in e.dc_plugs))
                plug_tesla.append(bool(DCPlug.TESLA in e.dc_plugs))
                n_evse += 1
                bank_running_power += e.max_power
                n_in_bank += 1
        bank_id += 1
        bank_power_max_val = bank,get('max_power', bank_running_power)
        bank_power_max.extend([bank_power_max_val for i in range(n_in_bank)])
    
    a = np.empty(n_evse, dtype=evse_)

    a[:]['bank_id'] = bank_ids
    a[:]['power'] = 0.
    a[:]['power_max'] = power_max
    a[:]['bank_power_max'] = bank_power_max
    a[:]['max_soc'] = max_soc
    a[:]['probability'] = probability
    a[:]['dc'] = dc
    a[:]['plug_chademo'] = plug_chademo
    a[:]['plug_combo'] = plug_combo
    a[:]['plug_tesla'] = plug_tesla

    return a

@nb.njit(cache=True)
def throttle_bank(bank: np.array, max_factor: float = .9):
    bank_total = bank[:, EVSE_POWER].sum()
    bank_max = bank[0, EVSE_BANK_POWER_MAX]

    scale_up = ((bank_max * max_factor) - bank_total) / bank_total
    scale_down = ((max_factor * bank_max) / bank_total)

    for i in range(bank.shape[0]):
        if bank[i, EVSE_POWER] > 0:
            if bank_total > bank_max:
                p = bank[i, EVSE_POWER] * scale_down
                bank[i, EVSE_POWER] = p
            elif bank_total < bank_max:
                p = bank[i, EVSE_POWER] + (bank[i, EVSE_POWER] * scale_up)
                bank[i, EVSE_POWER] = min(p, bank[i, EVSE_POWER_MAX])
    
    return bank

# Indexes in a vehicle
VEHICLE_ATTRIBUTE_LENGTH = 9

TYPE = 0
ICE_EFF = 1
ICE_GAL_PER_KWH = 2
EV_EFF = 3
EV_MAX_BATT = 4
AC_MAX = 5
DC_MAX = 6
DCPLUG_TYPE = 7
CONNECTED_EVSE_ID = 8

vehicle_ = np.dtype([
    ('type', np.int32, 1),
    ('ice_eff', np.float32, 1),
    ('ice_gal_kwh', np.float32, 1),
    ('ev_eff', np.float32, 1),
    ('ev_max_batt', np.float32, 1),
    ('ac_max', np.float32, 1),
    ('dc_max', np.float32, 1),
    ('dc_plug', np.int32, 1),
    ('home_evse_id', np.int64, 1),
    ('away_evse_id', np.int64, 1)
])

nb_vehicle_ = nb.from_dtype(vehicle_)

def make_fleet(
    vtypes: list,
    ice_effs: list,
    ice_gal_kwhs: list,
    ev_effs: list,
    ev_max_batts: list,
    ac_maxes: list, 
    dc_maxes: list,
    dcplugs: list):

    n = len(vtypes)
    a = np.empty(n, dtype=vehicle_)


    a[:]['type'] = vtypes
    a[:]['ice_eff'] = ice_effs
    a[:]['ice_gal_kwh'] = ice_gal_kwhs
    a[:]['ev_eff'] = ev_effs
    a[:]['ev_max_batt'] = ev_max_batts
    a[:]['ac_max'] = ac_maxes
    a[:]['dc_max'] = dc_maxes
    a[:]['dc_plug'] = dcplugs
    a[:]['home_evse_id'] = -1
    a[:]['away_evse_id'] = -1
    
    return a

def fleet_from_df(df: pd.DataFrame, powertrains: list,
                 probabilities: list):
        
        
    vnames = df.columns

    size = len(df.columns)

    pt_assignments = None

    if isinstance(powertrains, list):
        for pt in powertrains:
            if not isinstance(pt, Powertrain):
                raise TypeError()

        trains = powertrains
        num_trains = len(powertrains)
        num_probs = len(probabilities)
        probs = probabilities
        if not probabilities:
            probs = [float(1.0 / num_trains) for i in range(num_trains)]
        elif num_probs < num_trains:
            current_sum = sum(probs)
            if current_sum < 1:
                remaining = num_trains - num_probs
                distribution = (1 - current_sum) / remaining
                probs.extend([distribution for i in range(remaining)])
            else:
                trains = powertrains[:num_probs]
        elif num_trains < num_probs:
            probs = probabilities[:num_trains]

        pt_assignments = np.random.choice(
            range(num_trains),
            size=size,
            replace=True,
            p=np.array(probs)
        )

    vtypes = []
    ice_effs = []
    ice_gal_kwhs = []
    ev_effs = []
    ev_max_batts = []
    ac_maxes = [] 
    dc_maxes = []
    dcplugs = []

    for i in pt_assignments:
        pt = powertrains[i]
        pt: Powertrain
            
        vtypes.append(int(pt.ptype))
        ice_effs.append(pt.ice_eff)
        ice_gal_kwhs.append(pt.idle_fuel_consumption(1))
        ev_effs.append(pt.ev_eff)
        ev_max_batts.append(pt.batt_cap)
        ac_maxes.append(pt.ac_power)
        dc_maxes.append(pt.dc_power)
        dcplugs.append(pt.dc_plug)
        
    fleet = make_fleet(
        vtypes,
        ice_effs,
        ice_gal_kwhs,
        ev_effs,
        ev_max_batts,
        ac_maxes,
        dc_maxes,
        dcplugs
    )

    distance = np.array(df.values, dtype=np.float32)
    
    return distance, fleet

@nb.njit(cache=True)
def bank_enqueue(idx, vid, soc, fleet, queue, queue_soc):

    if fleet[vid]['type'] in (PHEV, BEV):
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
    queue[idx] = np.nan
    return queue

@nb.njit(cache=True)
def get_soc(vid, fleet, battery_nrg):
    out = 0.
    if fleet[vid]['ev_max_batt'] > 0:
        out = battery_nrg / fleet[vid]['ev_max_batt']
    return out

@nb.njit(cache=True)
def connect_evse(vid, soc, fleet, bank, away_bank = False):

    evse_ids = range(bank.shape[0])
    connected_power = 0

    for i in evse_ids:
        home_avail = True
        for j in range(fleet.shape[0]):
            if i == fleet[j]['home_evse_id']:
                home_avail = False
                break

        if ((home_avail and not away_bank)  or away_bank) and (random.random() <= bank[i]['probability']):
            if soc < bank[i]['max_soc']:
                # DCFC
                if bank[i]['dc'] and (fleet[vid]['dc_max'] > 0):
                    can_connect = False
                    if (fleet[vid]['dc_plug'] == DCPLUG_CHADEMO) and bank[i]['plug_chademo']:
                        can_connect = True
                    elif (fleet[vid]['dc_plug'] == DCPLUG_COMBO) and bank[i]['plug_combo']:
                        can_connect = True
                    elif (fleet[vid]['dc_plug'] == DCPLUG_TESLA) and bank[i]['plug_tesla']:
                        can_connect = True
                    
                    if can_connect:
                        connected_power = min(fleet[vid]['dc_max'], bank[i]['power_max'])
                        bank[i]['power'] = connected_power
                        if away_bank:
                            fleet[vid]['away_evse_id'] = i
                        else:
                            fleet[vid]['home_evse_id'] = i

                        break
                # AC
                else:
                    connected_power = min(fleet[vid]['ac_max'], bank[i]['power_max'])
                    bank[i]['power'] = connected_power
                    if away_bank:
                        fleet[vid]['away_evse_id'] = i
                    else:
                        fleet[vid]['home_evse_id'] = i
                    break
    
    return bool(connected_power > 0)

@nb.njit(cache=True)
def disconnect_evse(vid, fleet, bank, away_bank = False):
    eid = -1
    if away_bank:
        eid = fleet[vid]['away_evse_id']
    else:
        eid = fleet[vid]['home_evse_id']
    if eid >= 0:
        if not away_bank:
            bank[nb.uint64(eid)]['power'] = 0.
        if away_bank:
            fleet[vid]['away_evse_id'] = -1
        else:
            fleet[vid]['home_evse_id'] = -1          
    return eid

@nb.njit(cache=True)
def disconnect_completed(idx, battery_state, fleet, bank, away_bank = False):
    for vid in range(fleet.shape[0]):
        if idx > 0:
            nrg = battery_state[idx - 1, vid]
            eid = -1

            if away_bank:
                eid = fleet[vid]['away_evse_id']
            else:
                eid = fleet[vid]['home_evse_id']
            
            if eid >= 0:

                soc = get_soc(vid, fleet, nrg)
                if soc >= bank[nb.int64(eid)]['max_soc']:
                    disconnect_evse(vid, fleet, bank, away_bank)

@nb.njit
def charge(idx, battery_state, vid, fleet, eid, bank, min_per_interval):
    if nb.int64(eid) >= 0:
        power = bank[nb.int64(eid)]['power']
        potential = (min_per_interval / 60.0) * power
        max_nrg = fleet[vid]['ev_max_batt'] * bank[nb.int64(eid)]['max_soc']
        prev = fleet[vid]['ev_max_batt']
        if idx > 0:
            prev = battery_state[idx - 1, vid]
        nrg = min(prev+potential, max_nrg)
        battery_state[idx, vid] = nrg


@nb.njit(cache=True)
def charge_connected(idx, battery_state, fleet, home_bank, away_bank, min_per_interval):

    for i in range(fleet.shape[0]):
        home_eid = fleet[i]['home_evse_id']
        away_eid = fleet[i]['away_evse_id']
        if home_eid >= 0:
            charge(idx, battery_state, i, fleet, np.int64(home_eid), home_bank, min_per_interval)
        elif away_eid >= 0:
            charge(idx, battery_state, i, fleet, np.int64(away_eid), away_bank, min_per_interval)
    return battery_state

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
def stop_length_min(distance, idx, vid, min_per_interval):
    i = 0
    while (idx + i < distance.shape[0]) and (distance[idx+i, vid] == 0.):
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
def battery_req(distance, idle_load_kw, min_per_interval, vid, fleet):
    nrg = 0
    for idx in range(distance.shape[0]):
        # driving
        if distance[idx, vid] > 0:
            eff = fleet[vid]['ev_eff']
            if eff > 0:
                nrg += distance[idx, vid] / eff
        # idle
        elif distance[idx, vid] < 0:
            nrg += idle_load_kw * (min_per_interval / 60.0)
        idx += 1
    return nrg

@nb.njit(cache=True)
def try_defer_trips(
    idx,
    vid,
    fleet,
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
    if fleet[vid]['type'] == BEV:
        req = battery_nrg + 100
        nrg = battery_nrg * (1. - soc_buffer)
        end = distance.shape[0]
        stop = end
        i = idx
        while (i < end) and (nrg < req):

            stop = find_next_stop(distance, i, vid)
            stop_mins = stop_length_min(distance, stop, vid, min_per_interval)

            # We have an eligible stop!
            if (stop_mins >= away_thresh) or (stop_mins >= home_thresh) or mask[i]:
                req = battery_req(distance[i:stop,:], idle_load_kw, min_per_interval, vid, fleet)

                if nrg >= req:
                    break
                
                defer_to_target(distance[idx:stop, vid], deferred[idx:stop, vid])

            i = stop
            while distance[i, vid] == 0.:
                i += 1


@nb.njit(cache=True)
def drive(idx, distance, battery_state, fuel_use, fleet, idle_load_kw, min_per_interval):

    if idx < distance.shape[0]:
        for vid in range(fleet.shape[0]):

            ev_eff = fleet[vid]['ev_eff']
            ice_eff = fleet[vid]['ice_eff']
            ice_g_kwh = fleet[vid]['ice_gal_kwh']
            batt = 0
            if idx > 0:
                batt = battery_state[idx -1, vid]
            else:
                batt = fleet[vid]['ev_max_batt']


            batt_used = 0.
            fuel_used = 0.

            # driving
            if distance[idx, vid] > 0:
                d = distance[idx, vid]
                
                # Handle EV
                if (fleet[vid]['type'] in (BEV, PHEV)) and ev_eff > 0:
                    nrg_req = d / ev_eff
                    batt_used = min(batt, nrg_req)
                    d = d - (batt_used * ev_eff)
                
                if  (fleet[vid]['type'] in (ICEV, PHEV)) and ice_eff > 0:
                    fuel_used = d / ice_eff
            # idling
            elif distance[idx, vid] < 0:
                e = idle_load_kw * (min_per_interval / 60.)
                 # Handle EV
                if (fleet[vid]['type'] in (BEV, PHEV)) and ev_eff > 0:
                    batt_used = min(batt, e)
                    e = e - batt_used
                
                if  (fleet[vid]['type'] in (ICEV, PHEV)) and ice_g_kwh > 0:
                    fuel_used = e * ice_g_kwh
            

            battery_state[idx, vid] = batt - batt_used    
            fuel_use[idx, vid] = fuel_used

@nb.njit(cache=True)
def num_occupied_evse(fleet):
    n = 0
    for vid in range(fleet.shape[0]):
        eid = fleet[vid]['home_evse_id']
        if not (np.isnan(eid)):
            n += 1
    return n
        

@nb.njit(cache=True)
def evse_usage(idx, fleet, bank, demand, energy, occupancy, utilization, min_per_interval):

    max_power = bank[0]['bank_power_max']
    num_evse = bank.shape[0]
    total_power = 0.
    total_occupied = 0.
    
    for vid in range(fleet.shape[0]):
        eid = fleet[vid]['home_evse_id']
        if eid >= 0:
            power = bank[eid]['power']
            total_power += power
            total_occupied += 1
            energy[idx, eid] = power * (min_per_interval / 60.)
            demand[idx, eid] = power
    if num_evse > 0:
        occupancy[idx] = total_occupied / num_evse
    else:
        occupancy[idx] = 0
    if max_power > 0:
        utilization[idx] = total_power / max_power
    else:
        utilization[idx] = 0
    


@nb.njit(cache=True)
def simulation_loop(
    distance: np.array,
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

    nrows = distance.shape[0]
    nvehicles = fleet.shape[0]
    dshape = distance.shape
    nevse = home_bank.shape[0]

    fuel_use = np.zeros(dshape, dtype=np.float32)
    battery_state = np.zeros(dshape, dtype=np.float32)
    elec_demand = np.zeros((nrows, nevse), dtype=np.float32)
    elec_energy = np.zeros((nrows, nevse), dtype=np.float32)
    occupancy = np.zeros(nrows, dtype=np.float32)
    utilization = np.zeros(nrows, dtype=np.float32)
    deferred = np.zeros(dshape, dtype=np.float32)

    queue = np.full(nvehicles, np.nan, dtype=np.float32)

    for idx in range(nrows):

        # Disconnect and dequeue departing vehicles
        for vid in range(nvehicles):
            if not (distance[idx, vid] == 0.):
                bank_dequeue(queue, vid)
                disconnect_evse(vid, fleet, home_bank, False)
                disconnect_evse(vid, fleet, away_bank, True)
        
        # Disconnect completed vehicles
        if idx > 0:
            disconnect_completed(idx, battery_state, fleet, home_bank, False)
            disconnect_completed(idx, battery_state, fleet, away_bank, True)
        
        # Do drive and stop stuff
        if idx > 0 and idx < nrows:

            for vid in range(nvehicles):
            
                # Stop stuff
                if distance[idx, vid] == 0.:

                    # New stop
                    if not (distance[idx -1, vid] == 0.):
                        stop_time = stop_length_min(distance, idx, vid, interval_min)
                        soc = get_soc(vid, fleet, battery_state[idx - 1, vid])
                        
                        # Home stop
                        if stop_time >= home_thresh_min or home_mask[idx]:
                            
                            bank_enqueue(idx, vid, soc, fleet, queue, queue_soc)
                        
                        # Away stop
                        elif stop_time >= away_thresh_min:
                            connect_evse(vid, soc, fleet, away_bank, True)
                    
                    # About to depart
                    elif not (distance[idx + 1, vid] == 0.):
                        try_defer_trips(
                            idx,
                            vid,
                            fleet,
                            distance,
                            deferred,
                            home_mask,
                            battery_state[idx - 1, vid],
                            interval_min,
                            idle_load_kw,
                            away_thresh_min,
                            home_thresh_min,
                            soc_buffer
                        )
                    
        # Process queues and charge
        home_occupied = num_occupied_evse(fleet)
        while home_occupied < nevse:

            vid = pop_low_score(queue)

            if np.isnan(vid):
                break
            
            soc = get_soc(vid, fleet, battery_state[idx - 1, vid])
            
            connect_evse(vid, soc, fleet, home_bank, False )

            home_occupied += 1
        
        # Charge connected vehicles
        charge_connected(idx, battery_state, fleet, home_bank, away_bank, interval_min)

        evse_usage(idx, fleet, home_bank, elec_demand, elec_energy, occupancy, utilization, interval_min)

        drive(idx, distance, battery_state, fuel_use, fleet, idle_load_kw, interval_min)
    
    return (fuel_use, battery_state, deferred, elec_demand, elec_energy, occupancy, utilization)




def run_simulation(ds: DatasetInfo, sc: Scenario, storage_info: StorageInfo):
    """Runs a simulation with a given scenario by
    downloading data, running the simulation, and uploading results.

    Args:
        sc (Scenario): The scenario structure to use.
    
    Returns:
        A `SimulationResult` with all the relevant energy data.
    """
    st = DataHandler(storage_info)

    try:
        # First pull down the data
        
        df = st.read_data(ds.obj_path)

        df_index = df.index

        rows = len(df_index)

        interval_len = df.reset_index()['index'][:2].diff()[1].seconds / 60.0

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
        distance, fleet = fleet_from_df(df, sc.powertrains, sc.distribution)

        fleet_size = fleet.shape[0]

        home_banks = make_evse_banks(sc.home_banks, fleet_size)

        away_banks = make_evse_banks(sc.away_banks, fleet_size)

        num_banks = home_banks.shape[0]

        use_soc_queue = False

        output = simulation_loop(
            distance,
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

        evse_names = []
        for i in range(home_banks.shape[0]):
            if home_banks[i]['dc']:
                evse_names.append('ac{}-{}kW'.format(i, home_banks[i]['dc_max']))
            else:
                evse_names.append('ac{}-{}kW'.format(i, home_banks[i]['ac_max']))

        fuel_use, battery_state, deferred, elec_demand, elec_energy, occupancy, utilization = output

        vehicle_ids = df.columns.values.tolist()
        dfcols = df.columns

        fuel_df = pd.DataFrame(fuel_use, columns=dfcols, index=df_index)
        demand_df = pd.DataFrame(elec_demand, columns=evse_names, index=df_index)
        energy_df = pd.DataFrame(elec_energy, columns=evse_names, index=df_index)
        occupancy_df = pd.DataFrame({'occupancy': occupancy, 'utilization': utilization}, index=df_index)
        deferred_df = pd.DataFrame(deferred, columns=dfcols, index=df_index)
        battery_df = pd.DataFrame(battery_state, columns=dfcols, index=df_index)
        
        obj_base = 'results/' + 'simulations/' + sc.run_id + '/'

        dfs = [fuel_df, demand_df, battery_df, occupancy_df, deferred_df, energy_df]
        lbls = ['fuel', 'demand', 'battery', 'occupancy', 'deferred', 'energy']

        results = []

        for fr, nm in zip(dfs, lbls):
            
            result = st.upload_data(
                df=fr,
                obj_path=obj_base + nm + '/' +  ds.dataset_id,
                formats='json records csv',
            )
            if result:
                results.extend(result)

        
        

    except Exception as e:
        raise e

    finally:
        st.cleanup()

    return results






        
            






