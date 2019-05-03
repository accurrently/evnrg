import unittest

import pandas as pd
import numpy as np

from evnrg.powertrain import Powertrain
from evnrg.scenario import Scenario
from evnrg.evse import EVSEType
from evnrg.fuels import CARBOB_E10
from evnrg.plug import DCPlug
from evnrg import simulation_nb as sim

class TestSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create a sample distance df, and a test scenario
        """
        cls.dr = pd.date_range('2000-01-01 00:00:00', '2000-01-01 14:00:00', freq='5min')
        cls.df = pd.DataFrame(
            index=cls.dr
        )
        cls.df['ICEV-0'] = 0
        cls.df['PHEV-0'] = 0
        cls.df['PHEV-1'] = 0
        cls.df['BEV-0'] = 0
        cls.df['BEV-1'] = 0

        cls.df.loc['2000-01-01 07:00:00':'2000-01-01 08:00:00', 'ICEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 08:45:00':'2000-01-01 08:50:00', 'ICEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 09:00:00':'2000-01-01 09:30:00', 'ICEV-0'].iloc[:-1] = -1

        cls.df.loc['2000-01-01 06:30:00':'2000-01-01 07:00:00', 'PHEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 07:00:00':'2000-01-01 07:30:00', 'PHEV-0'].iloc[:-1] = -1
        cls.df.loc['2000-01-01 09:00:00':'2000-01-01 10:15:00', 'PHEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 13:00:00':'2000-01-01 14:00:00', 'PHEV-0'].iloc[:-1] = 1

        cls.df.loc['2000-01-01 05:30:00':'2000-01-01 07:00:00', 'PHEV-1'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 07:00:00':'2000-01-01 07:30:00', 'PHEV-1'].iloc[:-1] = -1
        cls.df.loc['2000-01-01 08:45:00':'2000-01-01 09:15:00', 'PHEV-1'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 12:00:00':'2000-01-01 12:30:00', 'PHEV-1'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 12:30:00':'2000-01-01 12:40:00', 'PHEV-1'].iloc[:-1] = -1

        cls.df.loc['2000-01-01 06:35:00':'2000-01-01 07:00:00', 'BEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 07:00:00':'2000-01-01 07:30:00', 'BEV-0'].iloc[:-1] = -1
        cls.df.loc['2000-01-01 08:15:00':'2000-01-01 9:00:00', 'BEV-0'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 11:00:00':'2000-01-01 11:20:00', 'BEV-0'].iloc[:-1] = 1

        cls.df.loc['2000-01-01 05:30:00':'2000-01-01 06:00:00', 'BEV-1'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 06:00:00':'2000-01-01 06:10:00', 'BEV-1'].iloc[:-1] = -1
        cls.df.loc['2000-01-01 08:30:00':'2000-01-01 09:50:00', 'BEV-1'].iloc[:-1] = 2
        cls.df.loc['2000-01-01 12:00:00':'2000-01-01 12:30:00', 'BEV-1'].iloc[:-1] = 1
        cls.df.loc['2000-01-01 12:30:00':'2000-01-01 12:40:00', 'BEV-1'].iloc[:-1] = -1

        cls.mask = pd.Series(
            [False for i in cls.dr],
            dtype=np.bool_,
            index=cls.dr
        )
        cls.mask.loc['2000-01-01 12:00:00':'2000-01-01 14:00:00'] = True
    
    @classmethod
    def tearDownClass(cls):
        cls.dr = None
        cls.df = None
        cls.mask = None

    def setUp(self):

        self.fleet = np.empty(5, dtype=sim.vehicle_)
        self.fleet[:]['type'] = [sim.ICEV, sim.PHEV, sim.PHEV, sim.BEV, sim.BEV]
        self.fleet[:]['ice_eff'] = [20., 30., 30., 0., 0.]
        self.fleet[:]['ice_gal_kwh'] = [
            0.1629004557450921, 0.1629004557450921, 0.1629004557450921, 0., 0.
        ]
        self.fleet[:]['ev_eff'] = [0, 2., 2., 2.1, 2.1]
        self.fleet[:]['ev_max_batt'] = [0., 20, 20, 60, 60]
        self.fleet[:]['ac_max'] = [0, 6, 6, 7, 7]
        self.fleet[:]['dc_max'] = [0, 50, 50, 50, 50]
        self.fleet[:]['dc_plug'] = [
            sim.DCPLUG_NONE,
            sim.DCPLUG_COMBO,
            sim.DCPLUG_COMBO,
            sim.DCPLUG_COMBO,
            sim.DCPLUG_COMBO
        ]
        self.fleet[:]['home_evse_id'] = -1
        self.fleet[:]['away_evse_id'] = -1

        self.home_bank = np.empty(3, dtype=sim.evse_)
        self.home_bank[:]['bank_id'] = 0
        self.home_bank[:]['power_max'] = [6.,7.,25.]
        self.home_bank[:]['bank_power_max'] = float(6+7+25)
        self.home_bank[:]['power'] = 0
        self.home_bank[:]['max_soc'] = [1.,1.,.8]
        self.home_bank[:]['probability'] = 1
        self.home_bank[:]['dc'] = [False, False, True]
        self.home_bank[:]['plug_chademo'] = [False, False, True]
        self.home_bank[:]['plug_combo'] = [False, False, True]
        self.home_bank[:]['plug_tesla'] = [False, False, False]


        self.away_bank = np.empty(1, dtype=sim.evse_)
        self.away_bank[:]['bank_id'] = 0
        self.away_bank[:]['power_max'] = 6
        self.away_bank[:]['bank_power_max'] = 6
        self.away_bank[:]['power'] = 0
        self.away_bank[:]['max_soc'] = 1
        self.away_bank[:]['probability'] = .2
        self.away_bank[:]['dc'] = False
        self.away_bank[:]['plug_chademo'] = False
        self.away_bank[:]['plug_combo'] = False
        self.away_bank[:]['plug_tesla'] = False
    
    def tearDown(self):
        self.fleet = None
        self.home_bank = None
        self.away_bank = None
    
    def test_make_evse_banks(self):

        hlist = [
            {
                'probability': 1,
                'evse': [
                    EVSEType(max_power=6., dc=False, min_=1, max_=0, pro_=0.),
                    EVSEType(max_power=7., dc=False, min_=1, max_=0, pro_=0.),
                    EVSEType(
                        max_power=25.,
                        dc=True,
                        dc_plugs=(sim.DCPLUG_CHADEMO, sim.DCPLUG_COMBO),
                        max_soc=.8,
                        min_=1, max_=0, pro_=0.
                    )
                ]
            }
        ]

        hb = sim.make_evse_banks(hlist, 5)
        self.assertEqual(self.home_bank.tolist(), hb.tolist())

        alist = [
            {
                'probability': .2,
                'evse': [
                    EVSEType(max_power=6., dc=False, min_=1, max_=0, pro_=0.)
                ]
            }
        ]

        ab = sim.make_evse_banks(alist, 5)

        self.assertEqual(self.away_bank.tolist(), ab.tolist())
    
    def test_make_fleet(self):

        fleet = sim.make_fleet(
            [sim.ICEV, sim.PHEV, sim.PHEV, sim.BEV, sim.BEV],
            [20., 30., 30., 0., 0.],
            [
                0.1629004557450921, 0.1629004557450921, 0.1629004557450921, 0., 0.
            ],
            [0, 2., 2., 2.1, 2.1],
            [0., 20, 20, 60, 60],
            [0, 6, 6, 7, 7],
            [0, 50, 50, 50, 50],
            [
                sim.DCPLUG_NONE,
                sim.DCPLUG_COMBO,
                sim.DCPLUG_COMBO,
                sim.DCPLUG_COMBO,
                sim.DCPLUG_COMBO
            ]
        )

        self.assertEqual(self.fleet.tolist(), fleet.tolist())
    
    def test_fleet_from_df(self):

        plist = [
            Powertrain(
                id = 'test-ICEV',
                ice_eff=20.,
                ev_eff=0.,
                batt_cap= 0.,
                ac_power= 0.,
                dc_power= 0.,
                dc_plug=DCPlug.NONE,
                ptype=sim.ICEV,
                fuel=CARBOB_E10,
                ice_alternator_eff = .21
            ),
            Powertrain(
                id = 'test-PHEV-0',
                ice_eff=30.,
                ev_eff=2.,
                batt_cap= 20.,
                ac_power= 6.,
                dc_power= 50.,
                dc_plug=DCPlug.COMBO,
                ptype=sim.PHEV,
                fuel=CARBOB_E10,
                ice_alternator_eff = .21
            ),
            Powertrain(
                id= 'test-PHEV-1',
                ice_eff=30.,
                ev_eff=2.,
                batt_cap= 20.,
                ac_power= 6.,
                dc_power= 50.,
                dc_plug=DCPlug.COMBO,
                ptype=sim.PHEV,
                fuel=CARBOB_E10,
                ice_alternator_eff = .21
            ),
            Powertrain(
                id= 'test-BEV-0',
                ice_eff=0.,
                ev_eff=2.1,
                batt_cap= 60.,
                ac_power= 7.,
                dc_power= 50.,
                dc_plug=DCPlug.COMBO,
                ptype=sim.BEV,
                fuel=None,
                ice_alternator_eff = 0
            ),
            Powertrain(
                id= 'test-BEV-1',
                ice_eff=0.,
                ev_eff=2.1,
                batt_cap= 60.,
                ac_power= 7.,
                dc_power= 50.,
                dc_plug=DCPlug.COMBO,
                ptype=sim.BEV,
                fuel=None,
                ice_alternator_eff = 0
            )
        ]

        distance, fleet = sim.fleet_from_df(__class__.df, plist)

        self.assertEqual(fleet.tolist(), self.fleet.tolist())
        self.assertEqual(__class__.df.values.tolist(), distance.tolist())

    def test_bank_enqueue(self):
        queue = np.array(
            [np.nan for i in range(len(__class__.df.columns))],
            dtype=np.float32,
        )

        queque = sim.bank_enqueue(2, 2, .5, self.fleet, queue, False)

        self.assertEqual(queue[2], 2)

        queue = sim.bank_enqueue(1, 4, .5, self.fleet, queue, True)

        self.assertEqual(queue[4], .5)
    
    def test_pop_low_score(self):

        z = [4, 2, 0]


        def good_pops(index_order, test_vals):
            queue = np.full(len(__class__.df.columns), np.nan, dtype=np.float32)
            for i, v in zip(index_order, test_vals):
                queue[i] = v
            
            y = []
            x = 0
            while x >= 0 and not np.isnan(x):
                x, queue = sim.pop_low_score(queue)
                if x >= 0:
                    y.append(x)
            
            self.assertEqual(y,index_order)

        
        good_pops(
            [4, 2, 0],
            [.3, .5, .7]
        )

        good_pops(
            [4, 2, 0],
            [13, 45, 72]
        )

        with self.assertRaises(TypeError):
            good_pops(
                [4, 2, 0],
                [
                    np.datetime64('2001-02-12'),
                    np.datetime64('2012-03-17'),
                    np.datetime64('2021-12-25')
                ]
            )

            good_pops(
                [4, 2, 0],
                [ 'a', 'b', 'c']
            )

            good_pops(
                [4, 2, 0],
                [ 'aa', 'cb', 'cc']
            )
    
    def test_bank_dequeue(self):

        q1 = np.full(len(__class__.df.columns), np.nan, dtype=np.float32)
        q2 = np.full(len(__class__.df.columns), np.nan, dtype=np.float32)


        q1[2] = 16
        sim.bank_dequeue(q1, 2)
        np.testing.assert_array_equal(q2, q1)

        q1[2] = 16
        with self.assertRaises(IndexError):
            sim.bank_dequeue(q1, 10)
            q1[13] = 9
    
    def test_get_soc(self):

        self.assertEqual(.5, sim.get_soc(4, self.fleet, 30.))
        self.assertEqual(1, sim.get_soc(4, self.fleet, 60.))
        self.assertEqual(0, sim.get_soc(4, self.fleet, 0))
        self.assertEqual(0, sim.get_soc(0, self.fleet, 30))

    def test_evse_connections(self):

        self.fleet, self.home_bank = sim.connect_evse(4, .5, self.fleet, self.home_bank)
        
        self.assertNotEqual(self.fleet[4]['home_evse_id'], -1)
        self.assertEqual(self.home_bank[0]['power'], 6)

        self.fleet, self.home_bank = sim.disconnect_evse(4, self.fleet, self.home_bank)

        self.assertEqual(self.fleet[4]['home_evse_id'], -1)
        self.assertEqual(self.home_bank[0]['power'], 0)

        self.fleet, self.home_bank = sim.connect_evse(4, 1, self.fleet, self.home_bank)

        self.assertEqual(self.fleet[4]['home_evse_id'],-1)
        self.assertEqual(self.home_bank[0]['power'], 0)
        

        
    
    def test_disconnect_completed(self):

        self.fleet, self.home_bank = sim.connect_evse(2, .5, self.fleet, self.home_bank)
        self.fleet, self.home_bank = sim.connect_evse(3, .5, self.fleet, self.home_bank)

        self.assertGreaterEqual(self.fleet[3]['home_evse_id'], 0)
        self.assertGreaterEqual(self.fleet[2]['home_evse_id'], 0)

        b = np.zeros(
            (2, len(__class__.df.columns)),
            dtype=np.float32
        )

        b[:, 3] = [60, np.nan]
        b[:, 2] = [15, np.nan]

        self.fleet, self.home_bank, self.away_bank = sim.disconnect_completed(b[0,:], self.fleet, self.home_bank, self.away_bank)

        self.assertEqual(self.fleet[3]['home_evse_id'], -1)
        self.assertGreaterEqual(self.fleet[2]['home_evse_id'], 0)
    
    def test_charge(self):

        b = np.zeros(
            (2, len(__class__.df.columns)),
            dtype=np.float32
        )
        b[:, 2] = [10, np.nan]
        b[:, 3] = [59.5, np.nan]
        

        self.fleet, self.home_bank = sim.connect_evse(2, .5, self.fleet, self.home_bank)
        self.fleet, self.home_bank = sim.connect_evse(3, .5, self.fleet, self.home_bank)

        self.assertGreaterEqual(self.fleet[2]['home_evse_id'], 0)
        self.assertGreaterEqual(self.fleet[3]['home_evse_id'], 0)
        
        
        p2 = self.home_bank[self.fleet[2]['home_evse_id']]['power']
        p3 = self.home_bank[self.fleet[3]['home_evse_id']]['power']

        s2 = self.home_bank[self.fleet[2]['home_evse_id']]['max_soc']
        s3 = self.home_bank[self.fleet[3]['home_evse_id']]['max_soc'] 

        b[1,2] = sim.charge(b[0,2], self.fleet[2]['ev_max_batt'], p2, s2, 60)
        b[1,3] = sim.charge(b[0,3], self.fleet[3]['ev_max_batt'], p3, s3, 60)

        self.assertEqual(b[1,2], 16)
        self.assertEqual(b[1,3], self.fleet[3]['ev_max_batt'])

    def test_charge_connected(self):
        b = np.zeros(
            (2, len(__class__.df.columns)),
            dtype=np.float32
        )
        b[:, 2] = [10, np.nan]
        b[:, 3] = [59.5, np.nan]
        

        self.fleet, self.home_bank = sim.connect_evse(2, .5, self.fleet, self.home_bank)
        self.fleet, self.home_bank = sim.connect_evse(3, .5, self.fleet, self.home_bank)

        
        self.assertGreaterEqual(self.fleet[2]['home_evse_id'], 0)
        self.assertGreaterEqual(self.fleet[3]['home_evse_id'], 0)

        self.assertGreaterEqual(self.home_bank[self.fleet[2]['home_evse_id']]['power'], 6)
        self.assertGreaterEqual(self.home_bank[self.fleet[2]['home_evse_id']]['power'], 6)

        b[1,:] = sim.charge_connected(b[0,:], self.fleet, self.home_bank, self.away_bank, 60)

        self.assertEqual(b[1,2], 16)
        self.assertEqual(b[1,3], self.fleet[3]['ev_max_batt'])
    
    def test_find_next_stop(self):

        df = __class__.df.copy()

        df: pd.DataFrame

        begin = df.index.get_loc(pd.Timestamp('2000-01-01 08:30:00'))
        
        end = df.index.get_loc(pd.Timestamp('2000-01-01 09:50:00'))

        stop = sim.find_next_stop(df.values, begin, 4)

        self.assertEqual(stop, end)

    def test_find_stop_end(self):

        df = __class__.df.copy()

        df: pd.DataFrame

        begin = df.index.get_loc(pd.Timestamp('2000-01-01 06:10:00'))
        
        end = df.index.get_loc(pd.Timestamp('2000-01-01 08:30:00'))

        stop = sim.find_stop_end(df.values, begin, 4)

        self.assertEqual(stop, end)
    
    def test_stop_length_min(self):

        df = __class__.df.copy()

        df: pd.DataFrame

        begin = df.index.get_loc(pd.Timestamp('2000-01-01 06:10:00'))

        # end at 0830, 1h20 min, or 140 min
        minutes = sim.stop_length_min(df.values[begin:,4], 5)

        self.assertEqual(minutes, 140)

    




















