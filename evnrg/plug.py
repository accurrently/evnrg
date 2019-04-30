import enum

class DCPlug(enum.IntEnum):
    """DCFC connector enumeration"""
    NONE = 0
    CHADEMO = enum.auto()
    COMBO = enum.auto()
    TESLA = enum.auto()
