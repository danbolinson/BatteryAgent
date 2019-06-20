import datetime
from requests import get
import numpy as np

apikey = 'RoFzaxXsygQJwsV52tTh3VIYkdf7dG8ETbZIlxSp'
getpage = '5b5bb82b5457a3a45d26b8fe'
eia = '14328'

class Tariff:
    '''Stores all information asssociatedw tih a specific tarriff. The class can calculate a bill for a biling period,
    for a specific moment of consumption based on tou rates, or based on a demand charge over a period.'''

    def __init__(self, api_url=None):
        if api_url is not None:
            page = 'https://api.openei.org/utility_rates'
            conn_string = page + "?"
            conn_string += "api_key={}".format(apikey)
            conn_string += "&format=json"
            conn_string += "&version=3"
            conn_string += "&getpage={}".format(getpage)
            conn_string += "&eia={}".format(eia)
            conn_string += "&detail=full"
            results = get(conn_string).json()
            self.tariff = results

        self.ENERGY_PEAK = 0.17425  # Actually part peak for A10S summer
        self.ENERGY_OFFPEAK = 0.14619  # off peak A10S summer
        self.DEMAND = 20.05  # A10S summer demand
        self.peak_period = [datetime.time(hour=9, minute=30), datetime.time(hour=18, minute=30)]

    def calculate_bill(self, bill_month, value='value'):
        '''Calculate the bill for the given bill month.
        Takes: Bill month should be given as a dataframe with value, timestamp(start) and duration_hrs columns.
        Returns: bill in dollars.'''
        demand_charge = self.calculate_demand_charge(bill_month, value)
        energy_charge = self.calculate_energy_charge(bill_month, value)

        return (demand_charge + energy_charge, (demand_charge, energy_charge))

    def calculate_energy_charge(self, bill_month, value='value'):
        '''Calculate the energy charge for the given bill month.
           Takes: Bill month should be given as a dataframe with value, timestamp(start) and duration_hrs columns.
           Returns: bill in dollars.'''
        return np.sum([self.calculate_energy_step(x, value) for i, x in bill_month.iterrows()])

    def calculate_energy_step(self, t, value='value'):
        '''Calculate the energy charge for the time step, a pandas series with start, duration_hrs, and value columns.
        Returns: Energy charge for that time step.'''
        # Check if peak-period and get energy rate
        if t.start.time() >= self.peak_period[0] and t.start.time() < self.peak_period[1]:
            rate = self.ENERGY_PEAK
        else:
            rate = self.ENERGY_OFFPEAK

        return t.duration_hrs * t[value] * rate

    def calculate_demand_charge(self, bill_month, value='value'):
        '''Calculate the demand charge for the given bill month.
        Takes: Bill month should be given as a dataframe with value, timestamp(start) and duration_hrs columns.
        Returns: bill in dollars.'''
        return self.DEMAND * max(bill_month[value])