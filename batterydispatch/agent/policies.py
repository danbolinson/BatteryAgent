from ..environment_assets.helpers import last_under, first_over
import operator
import numpy as np

def _get_max_dict_val(d, default_val = None):
    '''Takes in a dicctionary d, and returns the key associated with the maximum value.
    If two keys have equal values, one will be returned, without guarantee to which.'''
    try:
        return max(d.items(), key=operator.itemgetter(1))[0]
    except ValueError:
        if default_val is None:
            raise
        else:
            return default_val

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
    '''Takes the greedy action as determined by the state_action_values provided, except with probability eta.
    takes: eta (value between 0 and 1), and state_action_values (dictionary of dictionaries, {states: {actions}}'''

    eta = args['eta']
    s_a_values = args['state_action_values']


    if np.random.random() < eta:
        action = np.random.choice(actions)
    else:
        try:
            action = _get_max_dict_val(s_a_values[tuple(state)])
        except (ValueError, KeyError):
            action = 0

    return action

def choose_randomly(state, actions, period, args={}):
    return np.random.choice(actions)

def do_nothing(state, actions, period, args={}):
    if "do_nothing_action" not in args.keys():
        return 0
    else:
        return args['do_nothing_action']

