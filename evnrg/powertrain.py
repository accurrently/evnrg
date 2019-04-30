import enum
from typing import NamedTuple

from .fuels import Fuel, CARBOB_E10
from .plug import DCPlug

__all__ = [
    'PType',
    'Powertrain'
]

class PType(enum.IntEnum):
    """Constants that refer to the type of drive.

    Valid values are:
    * `ICEV`: Conventional combustion and hybrids
            (i.e. Honda Civic, Toyota Prius)
    * `PHEV`: Plug-in hybrids (i.e. Chevrolet Volt)
    * `BEV`: Battery electric vehicles (i.e. Tesla Model 3)
    """
    ICEV = enum.auto()
    PHEV = enum.auto()
    BEV = enum.auto()


class Powertrain(NamedTuple):
    """A `NamedTuple` that contains information about the powertrain of a vehicle.

    Since the powertrain technical info is all that's needed for the
    simulation; make and model information don't really matter.

    However, a code (optional) is useful to keep track of what powertrain
    has been used.

    Attributes:
        id (str): A short code useful for looking up the object in a database.
        ice_eff (float): Fuel efficiency of the powertrain in km/L.
        ev_eff (float): Electric efficiency in km/kWh.
        batt_cap (float): Energy capacity of the battery in kWh.
        ac_power (float): Maximum input AC power in kW.
        dc_power (float): Maximum input DC power in kW.
        dc_plug (int): Connector type for DCFC.
        ptype (int): Type of powertrain as represented by a PType.
            Values may be of `PType.ICEV`, `PType.PHEV`, or `PType.BEV`.
        fuel (Fuel): The fuel that this powertrain uses. May be `None`
            if `ptype` is `PType.BEV`.
    """
    id: str = ''
    ice_eff: float = 32.
    ev_eff: float = 0.
    batt_cap: float = 0.
    ac_power: float = 0.
    dc_power: float = 0.
    dc_plug: int = DCPlug.NONE
    ptype: int = PType.ICEV
    fuel: Fuel = CARBOB_E10
    ice_alternator_eff = .21

    @property
    def pev(self) -> bool:
        """Returns `True` if the powertrain is a plug-in(BEV or PHEV)"""
        return self.ptype in {PType.BEV, PType.PHEV}

    @property
    def bev(self) -> bool:
        """Returns `True` if the vehicle is a BEV"""
        return self.ptype == PType.BEV

    @property
    def has_ice(self) -> bool:
        """Returns `True` there's a fuel-powered engine (ICEV or PHEV)"""
        return self.ptype in [PType.ICEV, PType.PHEV]

    @property
    def phev(self) -> bool:
        """Returns `True` if the the powertrain is a PHEV."""
        return self.ptype == PType.PHEV

    @property
    def icev(self) -> bool:
        """Returns `True` if the powertrain is a conventional ICEV/HEV"""
        return self.ptype == PType.ICEV

    @property
    def dc_capable(self) -> bool:
        """Returns `True` if the powertrain can accept AC connections."""
        return self.dc_power > 0

    @property
    def ac_capable(self) -> bool:
        """Returns `True` if the powertrain can accept DC connections."""
        return self.ac_power > 0

    def energy_at_soc(self, soc) -> float:
        """Returns how how much energy is in the battery at a given SoC."""
        return self.batt_cap * max(0.0, min(soc, 1.0))

    def idle_fuel_consumption(self, load_kwh, use_gal = True) -> float:
        if self.ice_eff > 0 and self.ice_alternator_eff > 0:
            nrg_needed = load_kwh / self.ice_alternator_eff
            if use_gal:
                return nrg_needed / self.fuel.kWh_gal
            else:
                return nrg_needed / self.fuel.kWh_L
        return 0

