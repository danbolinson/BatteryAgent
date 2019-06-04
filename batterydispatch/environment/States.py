import numpy as np
import itertools
from .helpers import first_over, last_under

class States:
    '''Stores all information related to the state and handles conversion of a continuous varaible into a state. '''

    def __init__(self, hours=6, charge=4, load=10, demand=10):
        # Define the State variables S_ which capture the possible state spaces
        self.S_hours = np.arange(0, hours)  # Integer between 0 and 23 implying the hour of the day
        self.S_charges = np.arange(0, charge) / (
                    charge - 1)  # State of charge of the battery between 0 and 1, in 0.05 increments
        self.S_loads = np.arange(0, load) / (load - 1)  # load at time t
        self.S_demand = np.arange(0, demand) / (demand - 1)  # demand so far observed this month

        # Define the variables to hold the current state
        self.hour = None
        self.charge = None
        self.load = None
        self.demand = None

        # Define a boolean as to wehther the states have been fit to the System parameters
        self.fit = False

    def fit_load(self, max_load, min_load=0):
        self.S_loads = self.S_loads * (max_load - min_load) + min_load

    def fit_demand(self, max_load, min_load=None):
        if min_load is None:
            min_load = 0.25 * max_load

        self.S_demand = self.S_demand * (max_load - min_load) + min_load

    def fit_hours(self):
        self.S_hours = self.S_hours * 24 / len(self.S_hours)

    def fit_charge(self, max_charge, min_charge=0):
        self.S_charges = self.S_charges * (max_charge - min_charge) + min_charge

    def fit_parameters(self, max_load, max_charge, min_load=0, min_charge=0, min_demand=None):
        '''Scales the parameters from 0-1 to 0-given parameters.
        Whether this fitting has happened is monitored through the fit attribute and this is set to True.'''
        self.fit_hours()
        self.fit_load(max_load, min_load)
        self.fit_demand(max_load, min_demand)
        self.fit_charge(max_charge, min_charge)
        self.fit = True

    def as_tuple(self):
        '''Returns the State as a tuple (self.hour, self.charge, self.load, self.demand)'''
        return (self.hour, self.charge, self.load, self.demand)

    def set_state(self, hour, charge, load, demand, inplace=True):
        '''Sets the state for a given hour, charge, load, and demand. If inplace=False the internal sate is not changed.
        This allows assessing 'after states' by looking at what given environment values would mean in terms of state space.'''
        if self.fit == False:
            try:
                assert load <= 1
                assert hour <= 1
                assert charge <= 1
                assert demand <= 1
            except AssertionError:
                raise AssertionError(
                    "load, hour, demand, or charge > 1. Run fit_parameters() method or use fractional values.")

        shour = last_under(hour, self.S_hours)
        scharge = last_under(charge, self.S_charges)
        sload = first_over(load, self.S_loads)
        sdemand = first_over(demand, self.S_demand)

        if inplace:
            self.hour = shour
            self.charge = scharge
            self.load = sload
            self.demand = sdemand

        return (shour, scharge, sload, sdemand)

    def reset_state(self):
        '''Resets teh state variables hour, charge, load, demand to their default values.'''
        self.hour = self.S_hours[0]
        self.charge = self.S_charges[-1]
        self.load = self.S_loads[0]
        self.demand = self.S_demand[0]

    def list_all_states(self):
        '''Returns a list of the entire state space, i.e. all permutations of the state variables'''
        all_states = [l for l in itertools.product(*[self.S_hours, self.S_charges, self.S_loads, self.S_demand])]
        return all_states

    def __getitem__(self, index):
        if index == 'hour':
            return self.hour
        if index == 'charge':
            return self.charge
        if index == 'load':
            return self.load
        if index == 'demand':
            return self.demand
        else:
            raise KeyError("Invalid key {} for State. Allowed values are: ['hour', 'charge', 'load', 'demand']"
                           .format(index))

    def __str__(self):
        return str((self.as_tuple()))