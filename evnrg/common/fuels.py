from typing import NamedTuple
import enum

# Unit conversion constants
_GAL_PER_L = 0.264172
_LB_PER_KG = 2.20462
_BTU_PER_MJ = 947.82
_KWH_PER_MJ = 0.2778

class Fuel(NamedTuple):
    """Energy information for a given fuel"""
    MJ_L: float # MegaJoules
    ci: float # gCO2e/MJ

    @classmethod
    def from_Btu(cls, btu: float, ci: float):
        return Fuel(
            MJ_L = btu / _BTU_PER_MJ,
            ci = ci * _BTU_PER_MJ
        )
    
    @classmethod
    def from_mix(cls, fuel_tuples: list):

        mj = 0
        ci = 0
        pct_total = 0.   
        for fuel, pct in fuel_tuples:
            if not isinstance(fuel, Fuel):
                raise TypeError('All fuels must be of type Fuel')
            
            mj += fuel.MJ_L * pct
            ci += fuel.ci * pct
            pct_total += pct
        
        if not (pct_total == 1.):
            raise ValueError('Percentages do not add up to 1')

        return Fuel(mj, ci)
    
    @property
    def MJ_gal(self):
        return self.MJ_L * _GAL_PER_L
    
    @property
    def kWh_gal(self):
        return (self.MJ_L * _GAL_PER_L) / _KWH_PER_MJ
    
    @property
    def kWh_L(self):
        return self.MJ_L / _KWH_PER_MJ
    
    @property
    def kgCO2_gal(self):
        return ((self.ci * self.MJ_L) * _GAL_PER_L) / 1000.0
    
    @property
    def kgCO2_L(self):
        return (self.ci * self.MJ_L) / 1000.0

"""Fuel Information

The following Fuel information is provided by the
California Energy Commission and Air Resources Board.

Energy values taken from CA Energy Commission:
https://www.energy.ca.gov/2007publications/CEC-600-2007-002/CEC-600-2007-002-D.PDF

CI Values from CARB's LCFS:
https://www.arb.ca.gov/fuels/lcfs/fuelpathways/pathwaytable.htm
"""

# "Pure" fuels
CARBOB = Fuel(31.8, 99.78) # CA's "regular" gasoline - Conv. Petrol
DieselULS = Fuel(35.5, 102.01) # Ultra-low-sulfur diesel - Conv. Petrol
CNG = Fuel(.038, 79.46) # Compressed Natural Gas - North American Conventional
E100_CAC = Fuel(21.2, 53.49) # Pure ethanol - California Corn (T1R-1516)
BD100_TUCO = Fuel(32.6, 30.15) # Pure biodiesel - Texan Used Cooking Oil (T1N-1735)

# Fuel Mixtures
CARBOB_E10 = Fuel.from_mix(
    [(CARBOB, .9),
    (E100_CAC, .1)]
)

CARBOB_E85 = Fuel.from_mix(
    [(CARBOB, .15),
    (E100_CAC, .85)]
)

DieselULS_BD20 = Fuel.from_mix(
    [(DieselULS, .8),
    (BD100_TUCO, .2)]
)
