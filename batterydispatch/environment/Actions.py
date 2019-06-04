from .helpers import first_over, last_under
import numpy as np

class Actions:
    '''Class handling the actions associated with the battery dispatch RL problem.'''

    def __init__(self, charge_rates=2, charge_capacity=0.1):
        ''' list the battery dischage options. note that negative is charging, positive is discharging.'''
        charge_rates  # number of different charge values allowed, i.e. % of discharge rate
        charge_capacity  # fraction of battery capacity that can be charged in a given period
        self.options = np.arange(-charge_rates, charge_rates + 1) / charge_rates * charge_capacity

    def fit_actions(self, capacity):
        self.options = self.options * capacity

    def do_nothing(self):
        return first_over(0, self.options)

    def __len__(self):
        return len(self.options)

    def __getitem__(self, index):
        return self.options[index]
