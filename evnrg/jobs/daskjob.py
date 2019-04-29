from dask.distributed import progress
from dask.distributed import Client

import dask
import os
import uuid
from typing import List
import math
import logging

from evnrg.common.scenario import Scenario
from evnrg.storage.dataset import DatasetInfo
from evnrg.storage.datastorage import StorageInfo
from evnrg.simulation.simulation import run_simulation, SimulationResult
from evnrg.jobs.job_results import JobResults

class DaskJobRunner(object):

    __slots__ = (
        'scheduler_address',
        'client'
    )

    def __init__(self, scheduler_address = ''):

        self.scheduler_address = scheduler_address
        self.client = Client(self.scheduler_address)
    
    def run_simulations(self, scenarios: List[Scenario], datasets: List[DatasetInfo], 
                        storage_info: StorageInfo, print_client_info = True):

        client = self.client

        logging.info('Running Scenarios...')

        results = []

        for scenario in scenarios:
            for dataset in datasets:
                sim_result = client.submit(run_simulation, dataset, scenario, storage_info)
                results.append(sim_result)
          
        out = dask.compute(*results)
      
        return JobResults(out)
    
    
