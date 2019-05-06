import dask

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo
#from .job_results import JobResults
#from .simulation import run_simulation, SimulationResult
from .simulation_nb import run_simulation as nb_run_simulation

def simulation_job(self, scenarios: List[Scenario], datasets: List[DatasetInfo], 
                        storage_info: StorageInfo):

        results = []

        for scenario in scenarios:
            for dataset in datasets:
                                   
                sim_result = dask.delayed(nb_run_simulation)(dataset, scenario, storage_info)
                results.append(sim_result)
        return results