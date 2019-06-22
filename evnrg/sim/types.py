from typing import NamedTuple
from datetime import datetime

import pandas as pd
import numpy as np

class SimResult(NamedTuple):
    fuel: pd.DataFrame
    battery: pd.DataFrame
    deferred: pd.DataFrame
    demand: pd.DataFrame
    energy: pd.DataFrame
    summary: pd.DataFrame
    fleet: np.array
    home_bank: np.array
    trips: pd.DataFrame
    run_begin: datetime
    run_duration: float
