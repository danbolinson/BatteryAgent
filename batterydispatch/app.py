# import libraries
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree
from batterydispatch.environment.System import System
from batterydispatch.environment.Battery import Battery
from batterydispatch.agent.agents import ThresholdAgent, MonteCarloAgent, QAgent
from batterydispatch.agent.policies import do_nothing

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# Point a Path object at the GreenButton data
fdir = Path(os.path.dirname(__file__))
floc = "resources"
fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Large Usage.xml"
fpath = fdir / floc / fname
#fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Heavy Usage.xml"

# Set any policy parameters and choose a policy
TARGET_DEMAND = 8000
agent = ThresholdAgent(TARGET_DEMAND)


# Set up the system
system = System(fpath)
system.bus.add_battery(Battery(capacity=10000, power=2000))
system.set_actions(1, 0.2)
system.fit_states()
system.set_agent(agent)
system.initialize_A()

# Run first day
system.agent.set_policy(do_nothing)
_, (demand, energy) = system.tariff.calculate_bill(system.run_first_day())
default_reward = demand + energy * 30
default_reward

agent2 = QAgent()
system.set_agent(agent2)
system.agent.initialize_state_actions(system.state.list_all_states(), system.actions.options, -1*default_reward)
system.agent.default_SA_estimate = default_reward * -1

system.agent.set_greedy_policy(0.05)

hist = []

while True:
    grid_flow = system.run_first_day()
    peak_shave = max(grid_flow.load) - max(grid_flow.net_flow)
    demand = system.tariff.calculate_demand_charge(grid_flow, 'net_flow')
    energy = system.tariff.calculate_energy_charge(grid_flow, 'net_flow')
    orig_demand = system.tariff.calculate_demand_charge(grid_flow, 'load')
    orig_energy = system.tariff.calculate_energy_charge(grid_flow, 'load')

    reward = demand + energy * 30
    savings = orig_demand + orig_energy * 30 - reward
    print("Original demand: {}, new demand:{}, total reward: {}, savings: {}".format(max(grid_flow.load),
                                                                                     max(grid_flow.net_flow), reward,
                                                                                     savings))
    hist.append((peak_shave, reward, savings))
    if max(grid_flow.load) > max(grid_flow.net_flow):
        print("BOOM")
        #break

# Output the transition probability matrix A for the discharge Action and output to csv for review
DF_discharge = system.A_as_dataframe(2000)
DF_discharge.to_csv("D:/DF_discharge, A=2000.csv")