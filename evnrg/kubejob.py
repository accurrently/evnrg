from dask.distributed import progress
import dask
import os
import uuid
from typing import List
import math
import logging

from dask_k8 import DaskCluster

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo
from .simulation import run_simulation, SimulationResult
from .job_results import JobResults

from requirements import requirements

DEFAULT_DASK_SCHEDULER_YAML = """
containers:
  - image: daskdev/dask:latest
    command: ["sh", "-c"]
    args:
      - dask-scheduler --port 8786 --bokeh-port 8787
    imagePullPolicy: Always
    name: dask-scheduler
    ports:
      - containerPort: 8787
      - containerPort: 8786
    resources:
      requests:
          cpu: 1
          memory: 4G
"""

DEFAULT_DASK_WORKER_YAML = """
containers:
  - image: daskdev/dask:latest
    args: [dask-worker, $(DASK_SCHEDULER_ADDRESS), --nthreads, '1', --no-bokeh, --memory-limit, 3GB, --death-timeout, '60']
    imagePullPolicy: Always
    name: dask-worker
    env:
      - name: POD_IP
        valueFrom:
        fieldRef:
          fieldPath: status.podIP
      - name: POD_NAME
        valueFrom:
        fieldRef:
          fieldPath: metadata.name
      - name: EXTRA_PIP_PACKAGES
        value: 
      - name: EXTRA_CONDA_PACKAGES
        value: {conda_pkgs}
    resources:
      requests:
        cpu: 1
        memory: "3G"
      limits:
        cpu: 1
        memory: "3G"
""".format(
    conda_pkgs=' '.join(r for r in requirements)
)

class KubeJobCollection(object):
    """Runs a collection of scenarios and datasets through the simulation using Dask
    and Kubernetes.
    """

    __slots__ = (
        'storage_info',
        'sched_yaml',
        'worker_yaml',
        'namespace',
        'cluster_id',
        'n_workers',
        'client'
    )

    def __init__(self, storage_info: StorageInfo, num_workers: int,
                 namespace: str, cluster_id: str = 'evnrg-dask-{}'.format(uuid.uuid4().hex),
                 scheduler_pod_file: str = None, worker_pod_file: str = None,
                 scheduler_pod_spec: str = None, worker_pod_spec: str = None):

      if num_workers <= 0:
          raise ValueError('Number of workers must be >= 0!')
      
      # Initialize lists of stuff
      self.storage_info = storage_info,
      self.n_workers = num_workers
      
      self.sched_yaml = DEFAULT_DASK_SCHEDULER_YAML
      self.worker_yaml = DEFAULT_DASK_WORKER_YAML
      self.cluster_id = cluster_id
      self.namespace = namespace
      self.client = None

      if isinstance(scheduler_pod_file, str) and scheduler_pod_file:
          try:
              shed_file = open(scheduler_pod_file, 'r')
              self.sched_yaml = sched_file.read()
          except IOError:
              raise IOError('Schedule pod YAML file does not exist.')
      if isinstance(worker_pod_file, str) and worker_pod_file:
          try:
              work_file = open(worker_pod_file, 'r')
              self.worker_yaml = work_file.read()
          except IOError:
              raise IOError('Schedule pod YAML file does not exist.')
      
      if scheduler_pod_spec and isinstance(scheduler_pod_spec, str):
          self.sched_yaml = scheduler_pod_spec
      if worker_pod_spec and isinstance(worker_pod_spec, str):
          self.worker_yaml = worker_pod_spec
        
    def run_simulations(self, scenarios: List[Scenario], datasets: List[DatasetInfo], print_client_info = True):

      logging.info('Creating cluster object...')
      cluster = DaskCluster(
          namespace=self.namespace,
          cluster_id=self.cluster_id,
          worker_pod_spec=self.worker_yaml,
          scheduler_pod_spec=self.sched_yaml
      )
      out = []

      logging.info('Starting cluster...')
      with cluster:
          logging.info('Making Dask client...')
          client = cluster.make_dask_client()  # Waits for the scheduler to be started
          if print_client_info:
            print(client)
          cluster.scale(self.n_workers)

          results = []

          logging.info('Running Scenarios...')
          for scenario in scenarios:
              for dataset in datasets:
                  sim_result = client.submit(run_simulation, dataset, scenario, self.storage_info)
                  results.append(sim_result)
          
          out = results.result()
      
      return JobResults(out)
    
    @classmethod
    def calc_num_workers(cls, worker_mem: float, worker_cpu: float,
                       sched_mem: float, sched_cpu: float,
                       total_mem: float, total_cpu: float,
                       max_util: float = .85):
      
      mem_max = math.floor(((total_mem - sched_mem) * max_util) / worker_mem)
      cpu_max = math.floor(((total_cpu - sched_cpu) * max_util) / worker_cpu)

      return min(mem_max, cpu_max)

