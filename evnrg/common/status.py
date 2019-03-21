import enum


class Status(enum.IntEnum):
    NONE = 0

    DRIVING = 1

    STOPPED = INELIGIBLE = 2  # Ineligible to charge, but stopped
    AWAY_ELIGIBLE = 3      # Eligible for away charging due to length of stop
    HOME_ELIGIBLE = 4      # Eligible for home charging due to length of stop

    ARRIVED = enum.auto()
    IN_QUEUE = enum.auto()  # Currently in a charging queue
    CONNECTED = enum.auto()  # Currently connected to EVSE
    CONNECTED_V2G = enum.auto()  # Not implemented yet
    COMPLETED = enum.auto()
    DISCONNECTED = enum.auto()  # Disconnected
