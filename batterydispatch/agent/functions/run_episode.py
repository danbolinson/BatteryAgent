# This function runs the actual episodes, repeatedly, until policy converges.

from IPython.display import clear_output
# initial state
from batterydispatch.agent.agents import PolicyConvergedError

def run_episodes(env, agent, eps = 0, history = [], default_reward=None, random_charge=False, run_type='once'):
    possible_actions = list(env.action_mapping.keys())

    done = False
    i = 0

    over = False
    while not over:
        state = env.reset(random_charge=random_charge)

        eps += 1
        i += 1
        if i > 30:
            i = 0
            clear_output()

        while not done:
            action = agent.get_action(state, possible_actions, 0.25)
            # print(action)
            old_state = state.copy()
            state, reward, done, details = env.step(action)

            agent.observe_sars(old_state, action, reward, state)
        try:
            agent.end_episode(reward)
        except PolicyConvergedError:
            converged = True
            print("Converged!")
        try:
            new_demand = max(env.grid_flow.net_flow)
            orig_demand = max(env.grid_flow.load)
        except AttributeError:
            new_demand = "???"
            orig_demand = "???"

        done = False
        ran_once = True

        orig_reward = -1*(env.tariff.calculate_demand_charge(env.grid_flow, 'load') +
                        env.tariff.calculate_energy_charge(env.grid_flow, 'load') * 30)

        print(
            f"Current reward of {int(reward)} / {int(orig_reward)}, {new_demand} / {orig_demand}, patience={agent.patience_counter}")


        if run_type == "to_convergence":
            over = converged
        elif run_type == "once":
            over = ran_once
        # converged = agent.check_policy_convergence(False)
        history.append((eps, reward, new_demand, orig_reward, orig_demand))

    return reward