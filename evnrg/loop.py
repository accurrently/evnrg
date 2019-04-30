from typing import List

from .eligibility import EligibilityRules
from .status import Status
from .bank import Bank
from .vehicle import Vehicle


__all__ = []



def simulation_loop(
        vehicles: List[Vehicle],
        home_banks: List[Bank],
        away_banks: List[Bank],
        rules: EligibilityRules,
        min_per_interval: float,
        num_intervals: int,
        idle_load_kw: float = 0.
        ):

    for index in range(num_intervals):

        # Rewrite any trips that might not be possible
        for vehicle in vehicles:
            vehicle: Vehicle
            vehicle.attempt_defer_trips(rules, min_per_interval, idle_load_kw)
        
        # Disconnect all the vehicles that are leaving or done
        # Home banks
        for bank in home_banks:

            bank.dequeue_early_departures(index)
            bank.disconnect_completed_vehicles(index)
            bank.disconnect_departing_vehicles(index)
        # Away banks
        for bank in away_banks:
            bank.disconnect_completed_vehicles(index)
            bank.disconnect_completed_vehicles(index)
            bank.disconnect_departing_vehicles(index)            

        # Drive vehicles or enter queues
        for vehicle in vehicles:
            if index >= vehicle.idx:
                
                if vehicle.status == Status.HOME_ELIGIBLE:
                    for bank in home_banks:
                        if bank.enqueue_vehicle_prob(vehicle):
                            break
                elif vehicle.status == Status.AWAY_ELIGIBLE:
                    for bank in away_banks:
                        if bank.enqueue_vehicle_prob(vehicle):
                            break

        # Process the EVSE queue and charge
        for bank in home_banks:
            bank.process_queue()
            bank.charge_connected(min_per_interval, index)

        for bank in away_banks:
            bank.process_queue()
            bank.charge_connected(min_per_interval, index)

        for vehicle in vehicles:
            if index >= vehicle.idx:
                vehicle.advance_index(rules, min_per_interval, idle_load_kw)


    return vehicles, home_banks
