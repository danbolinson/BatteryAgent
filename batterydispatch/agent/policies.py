from ..environment.helpers import last_under, first_over

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
