import random
import numpy as np
import numba as nb

from .plug import DCPlug

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
    idx,
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
                drive_batt_used[idx] += batt_used
                d = d - max((batt_used * ev_eff), d)
            
            # Handle ICE                
            if  ice_eff > 0.:
                fuel_used = d / ice_eff
                drive_fuel_used[idx] += fuel_used
                drive_fuel_gwp[idx] += fuel_used * fuel_gwp
        # idling
        elif distance[vid] < 0.:
            e = idle_load_kw * (min_per_interval / 60.)
            # Handle EV
            if ev_eff > 0.:
                batt_used = min(batt, e)
                e = e - max(e, batt_used)
                idle_batt_used[idx] += batt_used
            # Handle ICE
            if  ice_g_kwh > 0.:
                fuel_used = e * ice_g_kwh
                idle_fuel_used[idx] += fuel_used
                idle_fuel_gwp[idx] += fuel_used * fuel_gwp
        
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

@nb.njit(cache=True)
def drive_usage(fleet, distance, battery_state):

    nvics = max(fleet.shape[0], 1)

    nstopped = 0.
    ndriving = 0.
    ncharging = 0.
    nidle = 0.
    stopped_battery = 0.
    total_battery = 0.

    
    for i in range(nvics):
        total_battery += battery_state[i]
        if fleet[i]['home_evse_id'] >= 0:
            ncharging += 1
        if distance[i] < 0:
            nidle += 1
        elif distance[i] > 0:
            ndriving += 1
        elif distance[i] == 0:
            nstopped +=1
            stopped_battery += battery_state[i]
    
    pct_driving = ndriving / nvics
    pct_idle = nidle / nvics
    pct_stopped = nstopped / nvics
    pct_charging = ncharging / nvics
    pct_batt_available = stopped_battery / total_battery
       

    return (
        pct_driving,
        pct_idle,
        pct_stopped,
        pct_charging,
        stopped_battery,
        pct_batt_available
    )
