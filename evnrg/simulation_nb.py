import random
import math
from typing import NamedTuple
import uuid

from timeit import default_timer as timer

import numba as nb
import numpy as np
import pandas as pd


from .scenario import Scenario
from .datastorage import DatasetInfo, StorageInfo, DataHandler, UploadResult
from .evse import EVSEType
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
        e: EVSEType
        for e in bank['evse']:
            copies = round(fleet_size * e.pro_)
            if e.max_ > 0:
                copies = int(min(e.max_, copies))
            copies = int(max(e.min_, copies))

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
        bank_power_max_val = bank.get('max_power', bank_running_power)
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
def isin_(x, arr):
    """
    This function only exists because Numba (stupidly?) doesn't support 
    the 'in' keyword or np.isin().
    """
    exists = False
    for i in range(arr.shape[0]):
        if arr[i] == x:
            exists = True
            break
    return exists

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
    ('away_evse_id', np.int64, 1),
    ('input_power', np.float32, 1),
    ('input_max_soc', np.float32, 1)
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
    a[:]['input_power'] = 0.
    a[:]['input_max_soc'] = 0.
    
    return a

def fleet_from_df(df: pd.DataFrame, powertrains: list,
                 probabilities: list = []):
        
        
    vnames = df.columns

    size = len(df.columns)

    pt_assignments = None

    # Explicit assignment
    if (not probabilities) and len(powertrains) == size:
        pt_assignments = [i for i in range(len(powertrains))]

    elif isinstance(powertrains, list):
        for pt in powertrains:
            if not isinstance(pt, Powertrain):
                raise TypeError()
        
        trains = powertrains
        num_trains = len(powertrains)
        num_probs = len(probabilities)
        probs = probabilities
        if not probabilities:
            if len(powertrains) < size:
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
                if (fleet[vid]['dc_plug'] == DCPLUG_CHADEMO) and bank[i]['plug_chademo']:
                    can_connect = True
                elif (fleet[vid]['dc_plug'] == DCPLUG_COMBO) and bank[i]['plug_combo']:
                    can_connect = True
                elif (fleet[vid]['dc_plug'] == DCPLUG_TESLA) and bank[i]['plug_tesla']:
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

    if idx > 0:
        for i in range(fleet.shape[0]):
            max_nrg = fleet[i]['ev_max_batt'] * fleet[i]['input_max_soc']
            new_nrg = battery_state[idx - 1, i] + ((float(min_per_interval) / 60.0) * fleet[i]['input_power'])
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
def drive(distance, batt_state, battery, fuel, fleet, idle_load_kw, min_per_interval):

    out = np.zeros((2,fleet.shape[0]), dtype=np.float32)


    for vid in range(fleet.shape[0]):

        ev_eff = fleet[vid]['ev_eff']
        ice_eff = fleet[vid]['ice_eff']
        ice_g_kwh = fleet[vid]['ice_gal_kwh']
        batt = batt_state[vid]

        batt_used = 0.
        fuel_used = 0.

        # driving
        if distance[vid] > 0:
            d = distance[vid]
            
            # Handle EV
            if ev_eff > 0:
                nrg_req = d / ev_eff
                batt_used = min(batt, nrg_req)
                d = d - (batt_used * ev_eff)
            
            # Handle ICE                
            if  ice_eff > 0:
                fuel_used = d / ice_eff
        # idling
        elif distance[vid] < 0:
            e = idle_load_kw * (min_per_interval / 60.)
                # Handle EV
            if ev_eff > 0:
                batt_used = min(batt, e)
                e = e - batt_used
            # Handle ICE
            if  ice_g_kwh > 0:
                fuel_used = e * ice_g_kwh
        

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
    queue_length = np.zeros(nrows, dtype=np.int64)
    connected_home_evse = np.full((nrows, nvehicles), -1, dtype=np.int64)

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

                # Stop stuff
                if curr_d == 0.:

                    # New stop
                    if not ( prev_d == 0.):
                        stop_time = stop_length_min(distance[idx:,vid], interval_min)
                        soc = get_soc(vid, fleet, battery_state[idx - 1, vid])
                        
                        # Home stop
                        if stop_time >= home_thresh_min or home_mask[idx]:
                            # Test to make sure we even have a home bank!
                            if home_bank.shape[0] > 0:
                                queue = bank_enqueue(idx, vid, soc, fleet, queue, queue_soc)
                            
                        
                        # Away stop
                        elif stop_time >= away_thresh_min:
                            connect_direct(vid, fleet, distance[idx-1], away_bank, away_bank = True)
                    
                    # About to depart
                    elif idx < nrows - 1:
                        # ^^ Required check to make sure we don't overrun the array.
                        # Remember that Numba will happily overrun without question.
                        if not (distance[idx + 1, vid] == 0.):
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
        
        # Process queue
        connect_from_queue(queue, fleet, bs, home_bank)
        queue_length[idx] = queue_size(queue)
        connected_home_evse[idx] = fleet[:]['home_evse_id']
        
        # Charge connected vehicles
        
        charge_connected(idx, battery_state, fleet, interval_min)
        drive(distance[idx], bs, battery_state[idx,:], fuel_use[idx,:], fleet, idle_load_kw, interval_min)


        # Record usage
        usage_info = evse_usage(fleet, home_bank, interval_min)
        elec_demand[idx,:] = usage_info[0,:]
        elec_energy[idx,:] = usage_info[1,:]
        occupancy[idx] = usage_info[2,:].sum() 
        
        util = 0

        if home_bank.shape[0] > 0:
            if home_bank[0]['bank_power_max'] > 0:
                util = usage_info[0,:].sum() / home_bank[0]['bank_power_max']
        
        utilization[idx] = util
    
    return (fuel_use, battery_state, deferred, elec_demand, elec_energy, occupancy, utilization, queue_length, connected_home_evse)

@nb.njit
def installed_capacity(bank):
    n_ac = 0
    n_dc = 0
    installed_capacity = 0
    for i in range(bank.shape[0]):
        if bank[i]['power_max'] > 0:
            installed_capacity += bank[i]['power_max']
        if bank[i]['dc']:
            n_dc += 1
        else:
            n_ac += 1
    return (n_ac, n_dc, installed_capacity)


def run_simulation(ds: DatasetInfo, sc: Scenario, storage_info: StorageInfo):
    """Runs a simulation with a given scenario by
    downloading data, running the simulation, and uploading results.

    Args:
        sc (Scenario): The scenario structure to use.
    
    Returns:
        A `SimulationResult` with all the relevant energy data.
    """

    sim_start = pd.Timestamp.now()

    
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
        away_banks[:]['power'] = away_banks[:]['power_max']

        num_banks = home_banks.shape[0]

        use_soc_queue = False

        timer_begin = timer()
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
        timer_end = timer()

        evse_names = []
        for i in range(home_banks.shape[0]):
            if home_banks[i]['dc']:
                evse_names.append('ac{}-{}kW'.format(i, home_banks[i]['power_max']))
            else:
                evse_names.append('ac{}-{}kW'.format(i, home_banks[i]['power_max']))

        fuel_use, battery_state, deferred, elec_demand, elec_energy, occupancy, utilization, queue_length, connected_home_evse = output

        vehicle_ids = df.columns.values.tolist()
        dfcols = df.columns

        fuel_df = pd.DataFrame(fuel_use, columns=dfcols, index=df_index)
        demand_df = pd.DataFrame(elec_demand, columns=evse_names, index=df_index)
        energy_df = pd.DataFrame(elec_energy, columns=evse_names, index=df_index)
        occupancy_df = pd.DataFrame({
            'occupancy': occupancy,
            'utilization': utilization,
            'queue_size': queue_length},
            index=df_index
        )
        deferred_df = pd.DataFrame(deferred, columns=dfcols, index=df_index)
        battery_df = pd.DataFrame(battery_state, columns=dfcols, index=df_index)
        home_evse_df = pd.DataFrame(connected_home_evse, columns=dfcols, index=df_index)
        
        obj_base = 'results/' + 'simulations/' + sc.run_id + '/'

        dfs = [fuel_df, demand_df, battery_df, occupancy_df, deferred_df, energy_df, home_evse_df]
        lbls = ['fuel', 'demand', 'battery', 'occupancy', 'deferred', 'energy', 'connected_evse']


        n_ac, n_dc, pow_cap = installed_capacity(home_banks)

        results =  {
            'run_start': sim_start,
            'scenario_id': sc.run_id,
            'fleet_id': ds.dataset_id,
            'ac_evse': n_ac,
            'dc_evse': n_dc,
            'installed_evse_capacity': pow_cap,
            'execution_time': pd.Timedelta(timer_end-timer_begin, 's'),
            'execution_time_sec': timer_end-timer_begin,
            'fleet_size': fleet_size,
            'rows': rows,
            'interval_length_min': interval_len
        }

        upload_uid = ''

        for fr, nm in zip(dfs, lbls):
            opath = obj_base + nm + '/' +  ds.dataset_id
            
            result = st.upload_data(
                df=fr,
                obj_path=opath,
                formats='json records csv',
            )

            if result:
                res: UploadResult
                for res in result:
                    results['upload_uid'] = res.uid
                    k = '{}_{}'.format(nm, res.filetype)
                    results[k] = res.remote_path
        
        

    except Exception as e:
        raise e

    finally:
        st.cleanup()

    return results






        
            






