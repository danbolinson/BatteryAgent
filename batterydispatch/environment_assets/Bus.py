from .Battery import Battery
import numpy as np

class Bus:
    '''Class handling the integration of the battery, the load, and the grid. Right now pretty dumb - just helps
    energy balance at any given period in time. The internal sate of the battery is maintained in teh Battery class.
    Batteries must be added using the add_battery method. Currently only one battery is supported.'''

    def __init__(self):
        self.grid_limit = np.inf
        self.battery = None

    def add_battery(self, battery):
        '''Add the given Battery object to the Bus.'''
        assert type(battery) == Battery
        self.battery = battery

    def calc_grid_flow(self, load, battery_action, time_period=1, affect_state=False):
        '''Calculates the flow to/from the grid for a given load and battery action, over a given time period in hours.
        Takes: load in kW, and battery_action in kW, and affect_state as a boolean as to whether to affect battery charge.
        Returns: Load on the grid in kW (+ve means drawing from the grid, -ve means outputting to the grid)'''
        battery_flow = self.battery.deploy(battery_action, time_period, affect_state)
        grid_flow = load - battery_flow
        if grid_flow > self.grid_limit:
            raise ValueError("The grid limit is exceeded by the input system parameters.")

        return grid_flow, battery_flow
