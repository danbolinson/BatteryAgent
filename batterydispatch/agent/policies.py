from ..environment.helpers import last_under, first_over
import operator
import numpy as np

def _get_max_dict_val(d):
    return max(d.items(), key=operator.itemgetter(1))[0]

def hard_limit(state, actions, period, demand_limit_dict):
    target = demand_limit_dict['target_demand']

    # try and keep demand below 4000
    if state['load'] > target:
        gap = state['load'] - target
        if gap > actions[-1]:
            action = actions[-1]
        else:
            action = first_over(gap, actions)
        if action > state['charge']:
            action = last_under(state['charge'], actions)  # COULD VIOLATE THE WHOLE ACTION BUCKET THING

    # charge between midnight and 5 am
    elif state['hour'] <= 5:
        action = actions[0]

    else:
        action = 0

    return action

def greedy_epsilon(state, actions, period, args):
    lr = args['learning_rate']
    s_a_values = args['state_action_values']


    if np.random.random() < lr:
        action = np.random.choice(actions.options)
    else:
        try:
            action = _get_max_dict_val(s_a_values[state.as_tuple()])
        except KeyError:
            action = 0

    return action

def choose_randomly(state, actions, period, args={}):
    return np.random.choice(actions.options)

def do_nothing(state, actions, period, args={}):
    return 0

