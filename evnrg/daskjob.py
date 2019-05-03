from dask.distributed import progress
from dask.distributed import Client

import dask
import os
import uuid
from typing import List
import math
import logging
from timeit import default_timer as timer
import pandas as pd

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo
#from .job_results import JobResults
#from .simulation import run_simulation, SimulationResult
from .simulation_nb import run_simulation as nb_run_simulation


class DaskJobRunner(object):

    __slots__ = (
        'scheduler_address',
        'client'
    )

    def __init__(self, scheduler_address = ''):

        self.scheduler_address = scheduler_address
        if not scheduler_address:
            self.client = Client()
            self.scheduler_address = 'localhost:8786'
        else:
            self.client = Client(self.scheduler_address)
    
    def run_simulations(self, scenarios: List[Scenario], datasets: List[DatasetInfo], 
                        storage_info: StorageInfo, use_numba = True):

        client = self.client

        logging.info('Running Scenarios...')

        #sim_f = run_simulation

        #if use_numba:
        print('Running simulations using Numba...')
        sim_f = nb_run_simulation


        results = []

        for scenario in scenarios:
            for dataset in datasets:
                                   
                sim_result = dask.delayed(sim_f)(dataset, scenario, storage_info)
                results.append(sim_result)
        
        
        records = dask.compute(*results)
      
        return pd.DataFrame(records)
    
    
