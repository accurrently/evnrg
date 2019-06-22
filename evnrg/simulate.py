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
from .evse import EVSEType, EVSEDef, Bank
from .powertrain import Powertrain, PType
from .plug import DCPlug
from .types import vehicle_, evse_, make_evse_banks, make_fleet
from .utils import isin_
from .data import fleet_from_df, banks_from_df, make_mask, write_data, load_data
from .datastorage import Storage, DatasetInfo, StorageInfo

from .masks import MaskRules

from .simulation.loop import sim_loop

def simulate(distance_df: pd.DataFrame, sc: Scenario):

    df = distance_df

    df_index = df.index

    rows = len(df_index)

    interval_len = df.index.to_series().diff().min().seconds / 60.0

    home_mask = sc.mask_rules.make_mask(df)

    fleet_size = len(df.columns)

    fleet = sc.fleet_def.make_fleet(fleet_size)

    home_banks = Bank.make_banks(sc.home_banks, fleet_size, False)

    away_banks = Bank.make_banks(sc.away_banks, fleet_size, True)

    away_banks[:]['power'] = away_banks[:]['power_max']

    num_banks = home_banks.shape[0]

    use_soc_queue = False

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
        sc.soc_buffer
    )

    return out