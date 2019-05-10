import pandas as pd
import numpy as np
from datetime import datetime

from ..scenario import Scenario
from ..datastorage import DatasetInfo, StorageInfo, DataHandler, UploadResult
from ..evse import EVSEType
from ..powertrain import Powertrain, PType
from ..plug import DCPlug

from .types import make_evse_banks, make_fleet


def load_data(ds: DatasetInfo, si: StorageInfo):
    st = DataHandler(si)

    df  = None

    try:
        df = st.read_data(ds.obj_path)
    except Exception as e:
        raise e

    finally:
        st.cleanup()
    
    return df

def write_data(
    df: pd.DataFrame,
    ds: DatasetInfo,
    si: StorageInfo,
    basepath: str,
    name: str = None,
    meta: dict = {},
    formats = 'records json csv'
    ):


    st = DataHandler(si)

    try:
        results = st.upload_data(
            df,
            obj_path=basepath,
            uid=name,
            formats = formats,
            meta=meta
        )
    except Exception as e:
        raise e
    finally:
        st.cleanup()
    
    return results

    

def banks_from_df(df: pd.DataFrame, sc: Scenario):
    size = len(df.columns)
    home_banks = make_evse_banks(sc.home_banks, size)
    away_banks = make_evse_banks(sc.away_banks, size)
    away_banks[:]['power'] = away_banks[:]['power_max']
    return (home_banks, away_banks)

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
    co2es = []

    for i in pt_assignments:
        pt = powertrains[i]
        pt: Powertrain
            
        vtypes.append(int(pt.ptype))
        ice_effs.append(pt.ice_eff)
        ice_gal_kwhs.append(pt.ice_gal_kWh)
        ev_effs.append(pt.ev_eff)
        ev_max_batts.append(pt.batt_cap)
        ac_maxes.append(pt.ac_power)
        dc_maxes.append(pt.dc_power)
        dcplugs.append(pt.dc_plug)
        if pt.fuel:
            co2es.append(pt.fuel.kgCO2_gal)
        else:
            co2es.append(0.)
        
    fleet = make_fleet(
        vtypes,
        ice_effs,
        ice_gal_kwhs,
        ev_effs,
        ev_max_batts,
        ac_maxes,
        dc_maxes,
        dcplugs,
        co2es
    )

    #distance = np.array(df.values, dtype=np.float32)
    
    return fleet

def make_mask(sc: Scenario, df: pd.DataFrame):
    mask = pd.Series(
            np.zeros(df.values.shape[0], dtype=np.bool_),
            index=df.index
        )

    for mask_rule in sc.home_mask_rules:
        if mask_rule.get('type') == 'time':
            begin = mask_rule.get('begin', '23:55')
            end = mask_rule.get('end', '00:00')
            mask[df.between_time(begin, end).index] = True
    
    return mask.values