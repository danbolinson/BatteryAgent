import gym

env = gym.make('gym_battery:battery-v0')
env.set_standard_system()

env.episode_type = env.episode_types[0]
env.episode_type

def dict_key_by_val(d, val):
    for k in d.keys():
        if d[k] == val:
            return k
    raise ValueError("value not found in dictionary")

act0 = dict_key_by_val(env.action_mapping, 0)

from batterydispatch.agent.agents import PrioritizedSweepingAgent, MonteCarloAgent, QAgent, DynaQAgent
from batterydispatch.agent.discretizers import Box_Discretizer

from batterydispatch.agent.policies import do_nothing
#MCAgent.initialize_state_actions(MCAgent.discretizer.list_all_states, env.action_space, default_val)
#agent = PrioritizedSweepingAgent(n_sweeps=3)
#agent = MonteCarloAgent()
agent = DynaQAgent()
agent.set_discretizer(Box_Discretizer(env.observation_space, N=[6, 4, 25, 25]))
agent.actions = env.action_space
agent.learning_rate = 0.15

done = False
state = env.reset()
i = 0
while not done:
    i+=1
    _,reward,done, details = env.step(act0)

from matplotlib import pyplot as plt
plt.plot(env.grid_flow.net_flow)
print(i)
print(reward)
default_reward = reward
#plt.show()

from gym.spaces import Box
import numpy as np

#env.observation_space = Box(np.array([0, 0, 1000, 1000]),np.array([24, env.bus.battery.capacity, 6000, 6500]))
agent.set_discretizer(Box_Discretizer(env.observation_space, N=[4, 3, 6, 6]))
agent.initialize_state_actions(new_default=default_reward, do_nothing_action = act0, do_nothing_bonus = 1)

from IPython.display import clear_output
# initial state
from batterydispatch.agent.agents import PolicyConvergedError


def run_to_convergence(random_charge=False):
    possible_actions = list(env.action_mapping.keys())
    converged = False
    done = False
    i = 0
    while not converged:
        state = env.reset(random_charge=random_charge)

        i += 1

        while not done:
            action = agent.get_action(state, possible_actions, 0.25)
            # print(action)
            old_state = state.copy()
            state, reward, done, details = env.step(action)

            if reward < -100000:
                print(reward)
                print(old_state)
                print(state)

            agent.observe_sars(old_state, action, reward, state)

        try:
            agent.end_episode(reward)
        except PolicyConvergedError:
            converged = True

        if agent.policy_converged_flag == True:
            converged = True

        try:
            new_demand = max(env.grid_flow.net_flow)
            orig_demand = max(env.grid_flow.load)
        except AttributeError:
            new_demand = "???"
            orig_demand = "???"

        done = False

        print(
            f"Current reward of {int(reward)} / {int(default_reward)}, {new_demand} / {orig_demand}, patience={agent.patience_counter}")
        # converged = agent.check_policy_convergence(False)
    print("Converged!")
    return i

agent.set_greedy_policy(eta=0.125)
agent.patience = 20
agent.subtype = 'off-policy'
agent.patience_counter = 0

import datetime

i = run_to_convergence(random_charge = False)

start = datetime.datetime.now()
print(start)
end = datetime.now()
print(end)
print("Took: {}".format(end-start))
print("episodes: {}".format(i))