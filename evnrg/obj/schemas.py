from typing import NamedTuple
import warnings

COL_STRLEN = 16
TIME_PRECISION = 'ms'
DEFAULT_FLOAT_PRECISION = 4

ITER_TYPES = (dict, set, tuple, list, str)

class ColumnKeyWarning(Warning):
    pass

class ColumnMap(NamedTuple):
    vehicle_id: str = 'vehicle_id'
    fleet_id: str = 'fleet_id'
    vclass: str = 'class'
    trip_begin: str='trip_begin'
    trip_end: str = 'trip_end'
    distance: str = 'distance',
    trip_h: str = 'trip_hours'
    stop_h: str = 'stop_hours'
    lat: str = 'latitude'
    lon: str = 'longitude'
    home_mask: str = 'home_mask'

    @property
    def schema(
        self,
        exclude: list = [],
        include: list = [],
        float_precision: int = DEFAULT_FLOAT_PRECISION,
        extra_cols: dict = {}):

        stype = 'U{}'.format(COL_STRLEN)
        ttype = 'datetime64[{}]'.format(TIME_PRECISION)
        ftype = 'f{}'.format(float_precision)

        d = {
            self.vehicle_id: stype,
            self.fleet_id: stype,
            self.vclass: stype,
            self.trip_begin: ttype,
            self.trip_end: ttype,
            self.distance: ftype,
            self.trip_h: ftype,
            self.stop_h: ftype,
            self.lat: ftype,
            self.lon: ftype,
            self.home_mask: bool
        }

        d.update(extra_cols)

        if exclude and isinstance(exclude, ITER_TYPES):
            for excl in exclude:
                try:
                    d.pop(excl)
                except KeyError:
                    warnings.warn(
                        ColumnKeyWarning(
                            '\'{}\' not found in columns.'.format(excl)
                        )
                    )
        
        if include and isinstance(exclude, ITER_TYPES):
            for incl in include:
                if not (incl in d.keys()):
                    warnings.warn(
                        ColumnKeyWarning(
                            '\'{}\' not found in columns.'.format(incl)
                        )
                    )

                for k in d.keys():
                    if not (k in include):
                        d.pop(k)
        
        return d

