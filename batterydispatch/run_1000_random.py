# import libraries
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree
from batterydispatch.environment.System import System
from batterydispatch.environment.Battery import Battery
from batterydispatch.agent.agents import ThresholdAgent, MonteCarloAgent

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# Point a Path object at the GreenButton data
fdir = Path(os.path.dirname(__file__))
floc = "resources"
fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Large Usage.xml"
fpath = fdir / floc / fname
#fname = "pge_electric_interval_data_2011-03-06_to_2012-04-06 A10S Med Business Heavy Usage.xml"

# Set any policy parameters and choose a policy
# TARGET_DEMAND = 8000
# agent = ThresholdAgent(TARGET_DEMAND)
agent = MonteCarloAgent()

# Set up the system
system = System(fpath)
system.bus.add_battery(Battery(capacity=10000, power=2000))
system.set_actions(1, 0.2)
system.fit_states()
system.set_agent(agent)
system.initialize_A()

# Run the first month and capture the grid_flow output
grid_flow = system.run_first_month(verbose=False)

import warnings
from tqdm import tqdm
warnings.simplefilter('ignore')

for i in tqdm(range(1000)):
    # Reinitialize A and run all months
    system.run_all_months()
