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

from .simulation_nb import run_simulation as nb_run_simulation

class DaskJobRunner(object):

    __slots__ = (
        'scheduler_address',
        'client',
        'jobs'
    )

    def __init__(self, scheduler_address = ''):

        self.scheduler_address = scheduler_address
        if not scheduler_address:
            self.client = Client()
            self.scheduler_address = 'localhost:8786'
        else:
            self.client = Client(self.scheduler_address)
        
        self.jobs = {}
    
    def add_job(self, name, jobf, args, kwargs):


        logging.info('Adding job: {}'.format(jobf))
        client = self.client

        self.jobs[name] = {
            'function': jobf.__name__,
            'future': jobf(*args, **kwargs),
            'result': None
        }
    
    def visualize_job(self, name):
        if self.jobs.get(name):
            dask.visualize(self.jobs['name']['future'])

    def run_jobs(self):
        client = self.client
        out = []
        for k, v in self.jobs.items():
            future = v['future']
            result = dask.compute(* future)
            self.jobs[k]['results'] = result
            out.append(result)
        return out

    
    def run_simulation(self, scenarios: List[Scenario], datasets: List[DatasetInfo], 
                        storage_info: StorageInfo, use_numba = True):

        client = self.client

        #sim_f = run_simulation

        #if use_numba:
        sim_f = nb_run_simulation


        results = []

        for scenario in scenarios:
            for dataset in datasets:
                                   
                sim_result = dask.delayed(sim_f)(dataset, scenario, storage_info)
                results.append(sim_result)
      
        return pd.DataFrame(records)