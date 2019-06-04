# import libraries
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree
from batterydispatch.environment.System import System
from batterydispatch.environment.Battery import Battery
from batterydispatch.agent.policies import hard_limit

# Point a Path object at the GreenButton data
floc = Path("./resources/")
fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Large Usage.xml"
#fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Heavy Usage.xml"

# Set any policy parameters and choose a policy
policy = hard_limit
TARGET_DEMAND = 8000
policy_args = {'target_demand': TARGET_DEMAND}

# Set up the system
system = System(floc / fname)
system.bus.add_battery(Battery(capacity=10000, power=2000))
system.set_actions(1, 0.2)
system.fit_states()
system.set_policy(policy, policy_args)
system.initialize_A()

# Run the first month and capture the grid_flow output
grid_flow = system.run_first_month(verbose=True)

# Reinitialize A and run all months
system.initialize_A()
system.run_all_months()

# Normalize A based on the learning (we lose the frequency of S --> S' this way - probably needs to change
system.normalize_A()

# Output the transition probability matrix A for the discharge Action and output to csv for review
DF_discharge = system.A_as_dataframe(2000)
DF_discharge.to_csv("D:/DF_discharge, A=2000.csv")