from typing import NamedTuple
import uuid

import pandas as pd
import numpy as np
import math

# Scenario information
from .scenario import Scenario
from .eligibility import EligibilityRules
from .fleet import Fleet
# Storage
from .datastorage import DatasetInfo, StorageInfo, DataHandler

# Objects
from .evse import EVSE, EVSEType
from .bank import Bank, QueueMode
from .status import Status

# The loop
from .loop import simulation_loop

__all__ = [
    'SimulationResult',
    'run_simulation'
]

class SimulationResult(NamedTuple):
    
    deferred: pd.DataFrame
    fuel_use: pd.DataFrame
    occupancy: pd.DataFrame
    demand: pd.DataFrame
    battery: pd.DataFrame
    scenario_id: str
    dataset_id: str
    dataset_path: str
    error: str = None
    
    def ok(self) -> bool:
        return not bool(self.error)


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

        rows = len(df.index)

        interval_len = df.reset_index()['index'][:2].diff()[1].seconds / 60.0

        # Basic rules creation for now

        mask_df = pd.DataFrame(
            np.zeros(df.values.shape, dtype=np.uint8),
            index=df.index
        )

        for mask_rule in sc.home_mask_rules:
            if mask_rule.get('type') == 'time':
                begin = mask_rule.get('begin', '23:55')
                end = mask_rule.get('end', '00:00')
                mask_df.loc[df.between_time(begin, end).index] = int(Status.HOME_ELIGIBLE)

        rules = EligibilityRules(
            away_threshold=round(sc.away_threshold_min/interval_len),
            home_threshold=round(sc.home_threshold_min/interval_len),
            mask=mask_df.values.astype(np.uint8)
        )

        # Create the fleet
        fleet = Fleet(df, sc.powertrains, sc.distribution, rules)

        # Get the banks up

        #away_banks = []
        #for bank_info in sc.away_banks:
        #    evse_list = []
        #    for evse_type in bank_info.get('evse'):
        #        evse_list.append(EVSE(evse_type))
        #    demand_a = np.zeros(rows, dtype=np.float32)
        #    occupancy_a = np.zeros(rows, dtype=np.uint8)
        #    bank = Bank(
        #        max_power=bank_info.get('max_power', 0.),
        #        capacity=bank_info.get('capacity', 0.),
        #        evse=evse_list,
        #        queue_probability=bank_info.get('probability', .2),
        #        queue_mode=bank_info.get('queue', QueueMode.DEFAULT),
        #        demand_profile=demand_a,
        #        occupancy_profile=occupancy_a,
        #        dynamic_size=True
        #    )
        #    away_banks.append(bank)

        away_banks = sc.make_away_banks(rows)
        
        #num_banks = 0
        #home_banks = []
        #for bank_info in sc.home_banks:
        #    evse_list = []
        #    for evse_type in bank_info.get('evse'):
        #        evse_list.append(EVSE(evse_type))
        #    bank = Bank(
        #        max_power=bank_info.get('max_power', 0.),
        #        capacity=bank_info.get('capacity', 0.),
        #        evse=evse_list,
        #        queue_probability=bank_info.get('probability', 1.),
        #        queue_mode=bank_info.get('queue', QueueMode.DEFAULT),
        #        demand_profile=np.zeros(rows, dtype=np.float32),
        #        occupancy_profile=np.zeros(rows, dtype=np.uint8),
        #        dynamic_size=False
        #    )
        #    home_banks.append(bank)
        #    num_banks += 1

        home_banks = sc.make_home_banks(fleet.size, rows)
        num_banks = len(home_banks)

        out_vehicles, out_banks = simulation_loop(
            fleet.vehicles,
            home_banks,
            away_banks,
            rules,
            interval_len,
            rows,
            sc.idle_load_kw
        )

        vehicle_ids = df.columns.values.tolist()
        fuel_df = pd.DataFrame(index=df.index)
        demand_df = pd.DataFrame(index=df.index)
        occupancy_df = pd.DataFrame(index=df.index)
        deferred_df = pd.DataFrame(index=df.index)
        battery_df = pd.DataFrame(index=df.index)

        for vehicle in out_vehicles:
            vid = str(vehicle.vid)
            fuel_df[vid] = vehicle.fuel_burned_a
            deferred_df[vid] = vehicle.deferred_a
            battery_df[vid] = vehicle.battery_a
        
        bank_i = 0
        for bank in out_banks:
            bank_col = 'bank_{}'.format(bank_i)
            demand_df[bank_col] = bank.demand_profile
            occupancy_df[bank_col] = (bank.occupancy_profile / bank.size) * 100.0 if bank.size > 0 else 0
            bank_i += 1
        if len(demand_df.columns) == 0:
            demand_df['bank_null'] = 0
        if len(occupancy_df.columns) == 0:
            demand_df['bank_null'] = 0
        
        obj_base = 'results/' + 'simulations/' + sc.run_id + '/'

        for fr, nm in zip([fuel_df, demand_df, battery_df, occupancy_df, deferred_df], ['fuel', 'demand', 'battery', 'occupancy', 'deferred']):
            
            st.upload_data(
                df=fr,
                obj_path=obj_base + nm + '/' +  ds.dataset_id,
                formats='parquet records csv',
            )
        
        

    except Exception as e:
        raise e

    finally:
        st.cleanup()

    return SimulationResult(
        scenario_id=sc.run_id,
        dataset_id=ds.dataset_id,
        dataset_path=ds.obj_path,
        deferred=deferred_df,
        fuel_use=fuel_df,
        occupancy=occupancy_df,
        demand=demand_df,
        battery=battery_df,
        error=None
    )    
    
    # return SimulationResult(errors=['Some error occurred.'])



