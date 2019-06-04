import numpy as np
import pandas as pd
from .IntervalReading import IntervalReading
from .Bus import Bus
from .Battery import Battery
from .Actions import Actions
from .States import States
from .Tariff import Tariff
from .helpers import last_under, first_over

class System:
    '''Class which monitors and executes the relationship between the agent and environment. Right now is dumb to the agent.
    The policy is currently established using the set_policy method. Episodes can be generated using teh run_First_month or
    run_all_months methods.'''

    def __init__(self, xml_location):
        self.load = IntervalReading(xml_location)
        self.actions = Actions()
        self.state = States()
        self.tariff = Tariff()
        self.bus = Bus()
        self.time = min(self.load.DF.start)
        self.policy = None

    def run_first_month(self, verbose=False):
        '''Run the first month as an episode; for testing purposes.'''
        df = next(self.load.get_month_generator())
        return self.run_episode(df, verbose)

    def run_all_months(self, verbose=False):
        '''Run each month as episodes.'''
        load_generator = self.load.get_month_generator()
        for df in load_generator:
            self.run_episode(df, verbose)

    def fit_states(self):
        '''This method fits the states and actions to the parameters of the bus and load in the system.
        This is helpful so everything is considered in kW (except battery capacity, in kWh).'''
        if self.bus.battery is None:
            battery_capacity = 0
        else:
            battery_capacity = self.bus.battery.capacity

        self.state.fit_parameters(self.load.get_max_load(), battery_capacity, self.load.get_min_load())

        self.actions.fit_actions(battery_capacity)

    def set_actions(self, charge_rates=2, charge_capacity=0.1):
        '''Sets teh number and size of actions based on the provied number of actions allowed and charge_capacity'''
        self.actions = Actions(charge_rates=charge_rates, charge_capacity=charge_capacity)

    def set_policy(self, policy, args=None):
        '''Sets teh policy that will be followed when running an episode.
        policy must be a function that takes in a State object, Action object, Period float parameters.
        State is the current state of the system.
        Action is the allowed actions within the system.
        Period is a float value indicating the duration of the step (in hours).
        Additional policy arguments can be stored and passed as a dictionary; they are passed as-is to function call.'''
        self.policy = policy
        self.policy_args = args

    def initialize_A(self):
        '''Initializes a dictionary of transition matrices A, frome ach posible state to each posssible state (matrix),
        for each possible action in actions (keys to the dictionary.
        {action: transition matrix}'''
        all_states = self.state.list_all_states()
        self.state_index = {state: ix for (ix, state) in enumerate(all_states)}
        self.sas_probability = {a: np.zeros((len(all_states), len(all_states))) for a in self.actions}
        self.all_states = all_states

    def normalize_A(self):
        '''Normalizes each row of each transition matrix by dividing by the sum of the row.'''
        for key in self.actions.options:
            self.sas_probability[key] = self.sas_probability[key] / np.sum(self.sas_probability[key], axis=1)[:,
                                                                      None]

    def A_as_dataframe(self, action):
        '''Returns the transition matrix as a dataframe for the given action action.'''
        return pd.DataFrame(self.sas_probability[action], index=self.all_states, columns=self.all_states)

    def run_episode(self, DF, verbose=False):
        '''
        Runs an episode using the month of load data passed to it as a DF.
        Best used as an internal function called by run_first_month or run_all_months.
        Initial episode conditions assume fully charged battery.
        '''
        self.state.reset_state()
        total_energy = 0
        demand = 0
        total_reward = 0

        imagined_demand = {}
        imagined_charge = {}
        first = True

        grid_flow = DF.copy(deep=True)
        grid_flow['battery_action'] = 0
        grid_flow['load'] = 0
        grid_flow['state_of_charge'] = 0

        # We will define each month as an episode
        for ix, t in DF.iterrows():
            # Capture the last state and update to the current state
            last_state = self.state.as_tuple()
            self.state.set_state(t.start.hour, self.bus.battery.charge, t.value, demand)

            # log what happened when we tried each action to build our transitional probabilities
            if not first:
                for a in self.actions:
                    imagined_state = self.state.set_state(t.start.hour,
                                                          imagined_charge[a],
                                                          t.value,
                                                          imagined_demand[a],
                                                          inplace=False)
                    self.sas_probability[a][self.state_index[last_state], self.state_index[imagined_state]] += 1

            period = t.duration_hrs

            # Try each action and examine the after-state to build the transitional probabilities
            for a in self.actions:
                imagined_flow, battery_flow = self.bus.calc_grid_flow(t.value, a, period, affect_state=False)
                imagined_demand[a] = max(demand, imagined_flow)
                imagined_charge[a] = self.state.charge + battery_flow * period

            # battery_action should be the discharge rate in kW.
            battery_action = self.policy(self.state, self.actions, period, self.policy_args)

            # Note calc-grid_flow takes care of battery discharge and affects the state of charge.
            net_flow, _ = self.bus.calc_grid_flow(t.value,
                                                  battery_action,
                                                  period,
                                                  affect_state=True)

            # IF the net_flow exceeds the demand, then update the episodic demand
            demand = max(net_flow, demand)

            grid_flow.loc[ix, 'value'] = net_flow
            grid_flow.loc[ix, 'load'] = t.value
            grid_flow.loc[ix, 'battery_action'] = battery_action
            grid_flow.loc[ix, 'state_of_charge'] = self.bus.battery.charge

            reward = -1 * self.tariff.calculate_energy_charge(grid_flow.loc[ix])
            total_reward += reward

            first = False

            if verbose:
                print("hour: {},  load(state): {}, soc:{}, load(actual): {}, demand(state): {}, battery action: {}"
                      .format(t.start.hour,
                              self.state.load,
                              self.bus.battery.charge,
                              t.value,
                              self.state.demand,
                              battery_action), end=" | ")
                print("net flow: {}, paid {}".format(net_flow, reward))

            # NEED TO FEED THE ACTION-REWARD BACK TO THE AGENT

        reward = self.tariff.calculate_demand_charge(grid_flow)
        total_reward += reward
        print("shaven demand: {} of peak {}".format(max(grid_flow.value), max(DF.value)))
        return grid_flow
        # NEED TO FEED FINAL REWARD BACK TO AGENT

