from dask.distributed import progress
from dask.distributed import Client

import dask
import os
import uuid
from typing import List
import math
import logging

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo
from .simulation import run_simulation, SimulationResult
from .numbaloop import run_simulation as nb_run_simulation
from .job_results import JobResults

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

        if use_numba:
            print('Running simulations using Numba...')
        else:
            print('Running simulations using plain ol\' Python...')


        results = []

        for scenario in scenarios:
            for dataset in datasets:
                if use_numba:
                    
                    sim_result = dask.delayed(nb_run_simulation)(dataset, scenario, storage_info)
                    results.append(sim_result)
                else:
                    sim_result = dask.delayed(run_simulation)(dataset, scenario, storage_info)
                    results.append(sim_result)
          
        out = dask.compute(*results)
      
        return out
    
    
