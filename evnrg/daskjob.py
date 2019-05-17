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
import numpy as np
from datetime import datetime
import seaborn as sns

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo

from .simulation_nb import run_simulation as nb_run_simulation

from .fuels import CA_MARGINAL_ELEC_CO2
from .types import (
    make_evse_banks,
    make_fleet
)
from .data import (
    load_data,
    write_data,
    write_data_iter,
    fleet_from_df,
    banks_from_df,
    make_mask
)
from .sim import (
    run_simulation as sim_loop_job
)
from .summarize import (
    energy_info,
    summarize_energy_info,
    add_datetime_cols,
    energy_pricing,
    get_col,
    sum_cols,
    make_co2e_df,
    add_time_cols,
    apply_lambda,
    add_id_cols
)
from .charts import (
    chart_demand,
    plot_facets,
    plot_demand,
    plot_bar
)

def delayed_apply(input_obj, func: callable, **kwargs):
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError('Obj must be a Pandas Series or DataFrame.')
    
    return obj.apply(func, **kwargs)




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
        records = dask.compute(*results)
        return pd.DataFrame(records)
    
    def run_full(self, scenarios: List[Scenario], datasets: List[DatasetInfo], 
                        storage_info: StorageInfo):
        
        client = self.client

        

        outputs = []
        
        rid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        si = storage_info

        sim_outputs = [
            'fuel',
            'battery',
            'deferred',
            'demand',
            'energy',
            'summary_info'
        ]

        price_data = []

        summarized_data = []

        demand_data = []
        total_co2e_data = []
        idle_co2e_data = []
        drive_co2e_data = []

        summary_sum_data = []
        summary_mean_data = []

        bbpath = 'results/' + rid + '/scenarios/'

        for sc in scenarios:
            sc: Scenario

            scenario_long = []
            #scenario_long = dask.delayed(scenario_long)

            scenario_short = []
            #scenario_short = dask.delayed(scenario_short)

            fleets_deferred = []

            sc_demand_data = []

            #sc = dask.delayed(scenario)
            idle_load = sc.idle_load_kw
            sid = sc.run_id

            bpath = bbpath + sc.run_id

            for ds in datasets:
                ds: DatasetInfo
               
                
                fid = ds.dataset_id                

                meta = dict(
                    scenario=sid,
                    fleet=fid,
                    idle_load=idle_load
                )

                sim_result = dask.delayed(pd.DataFrame, nout=6)

                sim_result = dask.delayed(sim_loop_job, nout=6)(
                    ds,
                    sc,
                    si
                )

                
                fuel_df = sim_result.fuel
                batt_df = sim_result.battery
                defer_df = sim_result.deferred
                demand_df = sim_result.demand
                nrg_df = sim_result.energy
                summary_df = sim_result.evse_info
                fleet = sim_result.fleet
                home_bank = sim_result.home_bank
                trips = sim_result.trips


                summary_df = dask.delayed(apply_lambda)(
                    summary_df,
                    'idle_batt_used',
                    'idle_batt_gwp',
                    lambda x: x * CA_MARGINAL_ELEC_CO2
                )

                summary_df = dask.delayed(apply_lambda)(
                    summary_df,
                    'drive_batt_used',
                    'drive_batt_gwp',
                    lambda x: x * CA_MARGINAL_ELEC_CO2
                )

                sim_results = [
                    fuel_df,
                    batt_df,
                    defer_df,
                    demand_df,
                    nrg_df,
                    summary_df
                ]

                for sr, so in zip(sim_results, sim_outputs):
                    for fmt in ('records', 'csv', 'json'):
                        outputs.append(
                            dask.delayed(write_data)(
                                df=sr,
                                ds=ds,
                                si=si,
                                name=so,
                                basepath=bpath,
                                meta=meta,
                                fmt=fmt
                            )
                        )
                
                

                nrg_nfo_df = dask.delayed(energy_info)(
                    fid,
                    sid,
                    fleet,
                    trips,
                    CA_MARGINAL_ELEC_CO2,
                    trips.index.to_series().diff().min().seconds / 60.0,
                    fuel_df.values,
                    batt_df.values,
                    demand_df.values
                )

                nrg_nfo_df = dask.delayed(add_datetime_cols)(
                    nrg_nfo_df
                )

                for fmt in ('records', 'csv', 'json'):
                    outputs.append(
                        dask.delayed(write_data)(
                            df=nrg_nfo_df,
                            ds=ds,
                            si=si,
                            name='energy_info',
                            basepath=bpath,
                            meta=meta,
                            fmt=fmt
                        )
                    )
                
                siminfo = pd.Series({
                    'fleet': fid,
                    'scenario': sid
                })

                summary_sums = dask.delayed(summary_df.sum)(axis=0)
                summary_sums = dask.delayed(summary_sums.append)(siminfo)
                summary_sum_data.append(summary_sums)

                summary_means = dask.delayed(summary_df.mean)(axis=0)
                summary_means = dask.delayed(summary_means.append)(siminfo)
                summary_mean_data.append(summary_means)

                demand_df = dask.delayed(add_id_cols)(
                    demand_df,
                    fid,
                    sid
                )

                demand_df = dask.delayed(add_time_cols)(demand_df)
                demand_df = dask.delayed(demand_df.reset_index)(drop=True)

                demand_data.append(
                    demand_df
                )

                sc_demand_data.append(demand_df)
                

                defer_totals = dask.delayed(defer_df.apply)(sum, axis=1)
                defer_totals = dask.delayed(defer_totals.rename)('deferred')

                #fleets_deferred.append(
                #    defer_totals
                #)

                scenario_long.append(nrg_nfo_df)
                scenario_long.append(defer_totals)
                scenario_long.append(summary_df)

                nrg_summary = dask.delayed(summarize_energy_info)(
                    df=nrg_nfo_df,
                    fname=fid,
                    scname=sid
                )

                scenario_short.append(nrg_summary)

                price_data.append(
                    dask.delayed(energy_pricing)(
                        nrg_summary,
                        dask.delayed(np.arange(0.10, 0.16, .02)),
                        dask.delayed(np.arange(2.5, 5, .05))
                    )
                )
                # End Fleet loop
            
            smeta = dask.delayed(dict)(
                    scenario=sid,
                    fleet=None,
                    idle_load=idle_load
            )

            sc_demand_df = dask.delayed(pd.concat)(
                sc_demand_data,
                axis=0
            )

            outputs.append(
                dask.delayed(write_data)(
                    sc_demand_df,
                    ds,
                    si,
                    basepath=bbpath,
                    name='demand',
                    fmt='csv'
                )
            )            
            # End Scenario loop

        summary_means_agg_df = dask.delayed(pd.DataFrame)(
            summary_mean_data
        )

        summary_sum_agg_df = dask.delayed(pd.DataFrame)(
            summary_sum_data
        )

        demand_full_df = dask.delayed(pd.concat)(
            demand_data,

        )

        outputs.append(
            dask.delayed(plot_demand)(
                demand_full_df,
                si,
                bbpath
            )
        )



        summary_df = dask.delayed(pd.DataFrame)(
            scenario_short
        )

        outputs.append(
            dask.delayed(plot_bar)(
                summary_df,
                si,
                basepath=bbpath,
                y='total_ghg_kgCO2',
                x='fleet',
                col='scenario',
                name='total_ghg'
            )
        )

        outputs.append(
            dask.delayed(plot_bar)(
                summary_df,
                si,
                basepath=bbpath,
                y='idle_ghg_kgCO2',
                x='fleet',
                col='scenario',
                name='idle_ghg'
            )
        )

        outputs.append(
            dask.delayed(write_data)(
                summary_df,
                ds,
                si,
                basepath=bbpath,
                name='overall-summary',
                fmt='csv'
            )
        )            

        output_rows = dask.compute(*outputs)

        return output_rows







                
