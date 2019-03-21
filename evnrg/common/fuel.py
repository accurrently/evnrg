from typing import NamedTuple

__all__ = [
    'Fuel',
    'GAS',
    'E10',
    'E85'
]


class Fuel(NamedTuple):
    """A `NamedTuple` that holds numeric values relevant to a fuel.

    Args:
        gwp (float): Gloabl warming potential (CO2e) in kg/L.
        id (:obj: `str`, optional): Short description (i.e. "E10").
        Defaults to an empty string ('').

    Attributes:
        gwp (float): Gloabl warming potential (CO2e) in kg/L.
        id (str): A short code describing the fuel.
    """
    gwp: float
    code: str = ''


# Here, we define a few defaults for good measure.
GAS = Fuel(code='GAS', gwp=8.78)  # Pure gasoline
E10 = Fuel(code='E10', gwp=8.58)  # E10 (standrd gas)
E85 = Fuel(code='E85', gwp=6.23)  # E85 Ethanaol (for Flex Fuel)
