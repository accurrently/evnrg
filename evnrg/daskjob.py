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
from datetime import datetime
import seaborn as sns

from .scenario import Scenario
from .dataset import DatasetInfo
from .datastorage import StorageInfo

from .simulation_nb import run_simulation as nb_run_simulation

from .fuels import CA_MARGINAL_ELEC_CO2
from .jobs.types import (
    make_evse_banks,
    make_fleet
)
from .jobs.data import (
    load_data,
    write_data,
    fleet_from_df,
    banks_from_df,
    make_mask
)
from .jobs.sim import (
    simulation_loop as sim_loop_job
)
from .jobs.summarize import (
    energy_info,
    summarize_energy_info,
    add_datetime_cols,
    energy_pricing
)
from .jobs.charts import (
    chart_demand,
    melt_and_plot_facets
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
        si = dask.delayed(storage_info)

        sim_outputs = [
            'fuel',
            'battery',
            'deferred',
            'demand',
            'energy',
            'evse_info'
        ]

        price_data = []

        summarized_data = []
        bbpath = 'results/' + rid + '/scenarios/'

        for scenario in scenarios:
            scenario: Scenario

            scenario_long = []
            scenario_long = dask.delayed(scenario_long)

            scenario_short = []
            scenario_short = dask.delayed(scenario_short)

            sc = dask.delayed(scenario)
            idle_load = sc.idle_load_kw
            sid = sc.run_id

            bpath = bbpath + scenario.run_id


            for dataset in datasets:
                dataset: DatasetInfo
                ds = dask.delayed(dataset)
                
                fid = ds.dataset_id                

                meta = dask.delayed(dict)(
                    scenario=sid,
                    fleet=fid,
                    idle_load=idle_load
                )

                

                trips = dask.delayed(load_data)(dataset, storage_info)

                fleet = dask.delayed(fleet_from_df)(
                    trips,
                    sc.powertrains, 
                    sc.distribution
                )

                fleet_size = fleet.shape[0]

                home_banks = dask.delayed(make_evse_banks)(
                    sc.home_banks,
                    fleet_size
                )

                away_banks = dask.delayed(make_evse_banks)(
                    sc.away_banks,
                    fleet_size
                )
                away_banks[:]['power'] = away_banks[:]['power_max']

                mask = dask.delayed(make_mask)(sc, trips)

                sim_result = sim_loop_job(
                    trips,
                    fleet,
                    home_banks,
                    away_banks,
                    mask,
                    sc.interval_min,
                    sc.home_thresh_min,
                    sc.away_thresh_min,
                    sc.idle_load_kw,
                    False,
                    sc.soc_deferment_buffer
                )

                for out_df, out_name in zip(sim_result, sim_outputs):
                    outputs.extend(
                        dask.delayed(write_data)(
                            out_df,
                            ds,
                            si,
                            name=dataset.dataset_id + '-' + out_name,
                            basepath=bpath,
                            meta = meta
                        )
                    )


                fuel_df, batt_df, defer_df, demand_df, nrg_df, evse_df = sim_result

                nrg_nfo_df = dask.delayed(energy_info)(
                    fid,
                    sid,
                    fleet,
                    trips,
                    CA_MARGINAL_ELEC_CO2,
                    trips.index.to_series.diff().min().seconds / 60.0,
                    fuel_df.values,
                    batt_df.values,
                    demand_df.values,

                )

                nrg_nfo_df = dask.delayed(add_datetime_cols)(
                    nrg_nfo_df
                )

                defer_totals = dask.delayed(defer_df.apply)(sum, axis=1)
                defer_totals = dask.delayed(defer_totals.rename)('deferred')

                fleets_deferred.append(
                    defer_totals
                )

                dask.delayed(scenario_long.append)(nrg_nfo_df)
                dask.delayed(scenario_long.append)(defer_totals)
                dask.delayed(scenario_long.append)(evse_df)

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
            
            smeta = dask.delayed(dict)(
                    scenario=sid,
                    fleet=None,
                    idle_load=idle_load
            )
            
            cost_df = dask.delayed(pd.concat)(
                price_data,
                axis=0,
                ignore_index=True
            )

            outputs.append(
                dask.delayed(melt_and_plot_facets)(
                    cost_df,
                    si,
                    id_vars=[
                        'fuel_price',
                        'elec_price',
                        'fleet'
                    ],
                    val_vars=[
                        'idle_fuel_cost',
                        'idle_elec_cost',
                        'drive_fuel_cost',
                        'drive_elec_cost',
                        'total_running_cost',
                    ],
                    val_name='cost_usd',
                    var_name='cost_type',
                    facet_opts={
                        'col': 'elec_price',
                        'row': 'cost_type',
                    },
                    map_func=sns.plot,
                    map_opts={
                        'hue': 'fleet',
                        'x': 'fuel_price',
                        'y': 'cost_usd'
                    },
                    basepath=bpath,
                    name='running_costs',
                    meta=smeta
                )
            )

        summary_df = dask.delayed(pd.concat)(
            scenario_short,
            axis=0
        )

        outputs.append(
            dask.delayed(melt_and_plot_facets)(
                summary_df,
                si,
                id_vars=[
                    'fleet',
                    'scenario'
                ],
                val_vars=[
                    'idle_ghg_kgCO2',
                    'drive_ghg_kgCO2',
                    'total_ghg_kgCO2',
                    'total_ghg_kgCO2'
                ],
                val_name='kg_CO2e',
                var_name='emissions',
                facet_opts={
                    'col': 'fleet',
                    'row': 'emissions',
                },
                map_func=sns.plot,
                map_opts={
                    'hue': 'scenario',
                    'x': 'scenario',
                    'y': 'kg_CO2e'
                },
                basepath=bbpath,
                name='emissions'
            )
        )

        outputs.extend(
            dask.delayed(write_data)(
                summary_df,
                ds,
                si,
                basepath=bbpath,
                name=dataset.dataset_id + '-' + 'summary',
                formats='csv'
            )
        )

            
            
        scenario_long_df = dask.delayed(pd.concat)(
            scenario_long,
            axis=1
        )

        outputs.append(
            dask.delayed(melt_and_plot_facets)(
                scenario_long_df,
                si,
                id_vars=[
                    'weekend_or_holiday'
                    'fleet',
                    'scenario',
                    'time',
                    
                ],
                val_vars=[
                    'home_demand_kW',
                    'stopped_battery_capacity_kWh',
                ],
                val_name='value',
                var_name='metric',
                facet_opts={
                    'col': ('weekend_or_holiday', 'metric'),
                    'row': 'scenario',
                },
                map_func=sns.lineplot,
                map_opts={
                    'hue': 'fleet',
                    'x': 'time',
                    'y': 'emissions'
                },
                basepath=bbpath,
                name='electrical_demand',
            )
        )

        outputs.extend(
            dask.delayed(write_data)(
                summary_df,
                ds,
                si,
                name=dataset.dataset_id + '-' + 'summary',
                basepath=bbpath,
                formats='csv'
            )
        )
    
        output_rows = dask.compute(*outputs)

        return pd.DataFrame(output_rows)







                
