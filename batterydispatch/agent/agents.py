from .policies import hard_limit, greedy_epsilon, choose_randomly
from .policies import _get_max_dict_val
import operator
from copy import deepcopy

from gym.spaces.box import Box, Space

import numpy as np
import random

from .discretizers import Discretizer, Box_Discretizer

def policy_changed(dict1, dict2):
    ''' Compares to state-action dictionaries {state:{action:value estimate}} and returns True if the
    greedy policy is the same, that is, the same action will be returned based on the argmax of state-value estimates.
    Returns false if either dictionary is missing a state that exists in the other.
    Takes: dict1 and dict2, both {state:{action:estimate}}
    Returns: True/False'''
    if set(dict1.keys()) !=  set(dict2.keys()):
        return False
    else:
        for s in dict1.keys():
            if _get_max_dict_val(dict1[s]) == _get_max_dict_val(dict2[s]):
                pass
            else:
                return False

    return True

# define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

class PolicyConvergedError(Exception):
    '''Raised when a policy has converged, typically after some amount of patience'''
    pass

class superAgent():
    '''superAgent provides the basic methods and properties that all (or at least many) agents will require.
    All agents should be subclasses of the superAgent class. superAgent should not be used.'''
    def __init__(self):
        self.policy = None
        self.name = 'superclass'
        self.S_A_values = {}
        self.S_A_frequency = {}
        self.history = []
        self.default_SA_estimate = 0
        self.patience = None
        self.patience_counter = 0
        self.policy_converged_flag = False
        self.past_S_A_values = deepcopy(self.S_A_values)
        self.discretizer = Discretizer()
        self.discrete = False
        self.actions = Space()
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def set_discretizer(self, discretizer):
        '''Set the discretizer used by the agent to convert a continuous state space intoo a discret state space.'''
        assert isinstance(discretizer, Discretizer)
        self.discrete = True
        self.discretizer = discretizer

    def discretize_space(self, space):
        '''Sends the space to the discretizer to be turned to discrete space.
        Takes: np.Array space object.
        Returns: np.Array of 'bin' in which conitniuous points fall..'''
        if not self.discrete:
            raise ValueError("The Agent is not expecting a discretized state space! Set a Discretizer.")

        return self.discretizer.discretize(space)

    def _follow_policy(self, state, actions, period, args={}):
        if self.policy is None:
            raise AssertionError("You must set a policy function before calling follow_policy.")

        if self.discrete:
            state = tuple(self.discretize_space(state))

        return self.policy(state, actions, period, self.policy_args)

    def get_action(self, state, actions=None, period=0.25):
        ''' get_action is the base method agents use to take an action, given a state.
        This should probably be overridden by whatever the agent needs to do.
        Takes: a state space, an action space (NEEDS REMOVAL), and the length of the step in hours (NEEDS REMOVAL)'''
        return self._follow_policy(state, actions, period)

    def check_policy_convergence(self, increment_patience=True):
        '''Checks to see if the policy has converged, by comparing S_A_values and past_S_A_values.
        Raisies an assertion error if patience parameter hasn't been set.
        Raises a PolicyConvergedError if the policy has in fact converged for the patience set.
        Takes: Whether or not to increment hte policy convergence counter.
        Returns: None.'''
        # End early
        assert self.patience is not None

        if policy_changed(self.S_A_values, self.past_S_A_values):
            self.patience_counter += increment_patience
        else:
            self.patience_counter = int(increment_patience)

        if self.patience_counter > self.patience:
            raise PolicyConvergedError

        self.past_S_A_values = deepcopy(self.S_A_values)

    def set_greedy_policy(self, eta=0.2):
        '''Sets the greedy policy as the policy the agenet will follow, with epsilon given.
        '''
        args = {
            'eta': eta,
            'state_action_values': self.S_A_values
        }

        self.set_policy(greedy_epsilon, args)

    def collect_reward(self, reward, state, action=None):
        # This takes a reward for a specific state or state-action
        pass

    def set_policy(self, policy, policy_args={}):
        self.policy = policy
        self.policy_args = policy_args

    def end_episode(self, reward=0):
        pass


    def observe_sars(self, state, action, reward, next_state):
        pass

    def initialize_state_actions(self, new_default=0, do_nothing_action=0, do_nothing_bonus=0):
        '''Initializes the S_A Tables for values and frequencies based on the states and actions known to the agent.'''
        if not isinstance(self.discretizer, Box_Discretizer):
            raise NotImplementedError("initializing state space is only set up for 1D Box state space")

        self.default_SA_estimate = new_default

        states = self.discretizer.list_all_states()
        np_actions = np.arange(self.actions.n)

        for s in states:
            s = tuple(s)
            for a in np_actions:
                self.S_A_values.setdefault(s, {})[a] = self.default_SA_estimate
                self.S_A_frequency.setdefault(s, {})[a] = 0

                if do_nothing_action is not None and a == do_nothing_action:
                    self.S_A_values[s][a] += do_nothing_bonus  # Break ties in favor of do nothing

class PrioritizedSweepingAgent(superAgent):

    def __init__(self, policy=choose_randomly, args={}, learning_rate = 0.1, patience=None, n_sweeps=5, sensitivity = 100):
        super().__init__()
        self.name = "Prioritized Sweeping Agent"
        self.set_policy(policy, args)
        self.learning_rate = learning_rate
        self.patience = patience
        self.P_queue = []
        self.n_sweeps = n_sweeps
        self.sweep_depth = 150
        self.sensitivity = sensitivity
        self.transition_table = np.array([])
        self.reward_table = {}
        self.x_lookup = {}
        self.y_lookup = {}
        self.state_ix_lookup = {}
        self.state_list = []

    def initialize_state_actions(self, default=0, do_nothing_action=0, do_nothing_bonus=0):
        '''Extends the base Agent classes initialize_state_actions to include details on the transition and reward
        model used by the algorithm'''
        super().initialize_state_actions(default, do_nothing_action, do_nothing_bonus)

        all_states = self.discretizer.list_all_states()
        all_states.append('terminal')
        self.state_list = all_states

        all_actions = np.arange(self.actions.n)

        all_sa = [(s, a) for a in all_actions for s in all_states]

        self.state_action_list = all_sa
        self.x_lookup = {(state, action): i for i, (state,action) in enumerate(all_sa)}
        self.y_lookup = {state: i for i, state in enumerate(all_states)}
        self.state_ix_lookup = {self.y_lookup[k]: k for k in self.y_lookup}
        self.sa_ix_lookup = {self.x_lookup[k]: k for k in self.x_lookup}
        self.transition_table = np.zeros((len(self.x_lookup), len(self.y_lookup)))
        self.reward_table = np.zeros(len(all_states))


    def observe_sars(self, state, action, reward, next_state):

        def insert_p(P):
            if P > self.sensitivity:
                if self.P_queue == []:
                    self.P_queue.append(((state, action), P))
                else:
                    for sa, P_sa in self.P_queue:
                        if P > P_sa:
                            self.P_queue.insert(self.P_queue.index((sa, P_sa)), ((state, action), P))
                            break

        # Discretize the space
        state = tuple(self.discretize_space(state))
        if next_state == 'terminal':
            pass
        else:
            next_state = tuple(self.discretize_space(next_state))

        # Update the state-action-state transition
        self.transition_table[self.x_lookup[(state, action)], self.y_lookup[next_state]] += 1

        try:
            self.reward_table[(state, action, next_state)] += \
                self.learning_rate * (reward - self.reward_table[state, action, next_state])
        except KeyError:
            self.reward_table[(state, action, next_state)] = reward

        def get_greedy_value(state, default=np.nan):
            'gets the greedy action value for the gien state'
            if state is not "terminal":
                greedy_next_action = _get_max_dict_val(self.S_A_values[state])
                return self.S_A_values[state][greedy_next_action]
            else:
                return default

        # Get the change and priority of the change
        greedy_value = get_greedy_value(next_state, default=0)
        P = abs(reward + greedy_value - self.S_A_values[state][action])
        insert_p(P)
        self.S_A_values[state][action] += \
            self.learning_rate * (reward + greedy_value - self.S_A_values[state][action])

        if abs(reward)>1000:
            print("end")

        # Do the planning step by emptying P_queue

        for i in range(self.n_sweeps):
            if self.P_queue == []:
                break

            (state, action), _ = self.P_queue.pop(0)
            #reward = self.reward_table[(state, action, ) STOPPED HERE, need to reconfigure so it correctly gets reward per hte new dict structure]
            reward = 0

            p_sas = self.transition_table[self.x_lookup[(state, action)], :]
            # Choose randomly from the next states based on their probability of being visited but ignoring terminal
            p_sas = p_sas[:-1] / np.sum(p_sas[:-1])

            greedy_value = np.sum([p_sas[ix] * get_greedy_value(self.state_ix_lookup[int(ix)]) for ix in np.nditer(np.nonzero(p_sas))])

            if greedy_value != np.nan:
                self.S_A_values[state][action] += \
                                        self.learning_rate * (reward + greedy_value - self.S_A_values[state][action])

            lead_to_s = state
            greedy_next_action = _get_max_dict_val(self.S_A_values[lead_to_s])

            # Next we choose N actions to chase back, to limit execution time, by choosing based on probability of ending up there
            likelihood = self.transition_table[:, self.y_lookup[lead_to_s]] \
                         / np.sum(self.transition_table[:, self.y_lookup[lead_to_s]])
            nonzero = np.nonzero(self.transition_table[:, self.y_lookup[lead_to_s]])[0]
            if len(nonzero) <= self.sweep_depth:
                selection = nonzero
            else:
                selection = tuple(np.random.choice(np.arange(self.transition_table.shape[0]),
                                         size=self.sweep_depth,
                                         replace=False,
                                         p=likelihood))
            for ix in selection:
                #sa_cnt = self.transition_table[ix, self.y_lookup[lead_to_s]]

                # Prioritize by taking top N only, randomply

                (state, action) = self.sa_ix_lookup[int(ix)]
                reward = self.reward_table[self.y_lookup[state]]
                P = abs(reward + self.S_A_values[lead_to_s][greedy_next_action] - self.S_A_values[state][action])
                insert_p(P)

    def end_episode(self, reward=0):
        if self.patience is not None:
            try:
                super().check_policy_convergence()
            except PolicyConvergedError:
                self.policy_converged_flag = True

class QAgent(superAgent):

    def __init__(self, policy=choose_randomly, args={}, epsilon = 0.1, alpha = 0.15, patience=None):
        super().__init__()
        self.name = "Q-Learning Agent"
        self.set_policy(policy, args)
        self.learning_rate = alpha
        self.patience=patience

    def get_action(self, state, actions, period):
        # This manages all aspects of the Q-learning algorithm including bootstrapping of the reward.
        # First, the last step is handled.
        # Then, the next action is selected.

        action = super()._follow_policy(state, actions, period)
        return action

    def end_episode(self, reward):
        # reset history
        self.history = []

        # Check if the policy has converged if a patience has been set
        if self.patience is not None:
            super().check_policy_convergence()

        self.past_S_A_values = deepcopy(self.S_A_values.copy())

    def observe_sars(self, state, action, reward, next_state):
        if (next_state == self.terminal_state).all():
            greedy_action_estimate = 0
        else:
            next_state = tuple(self.discretize_space(next_state))
            greedy_action = _get_max_dict_val(self.S_A_values[next_state], self.default_SA_estimate)
            greedy_action_estimate = self.S_A_values[next_state][greedy_action]

        state = tuple(self.discretize_space(state))
        self.S_A_values[state][action] += self.learning_rate*(reward + greedy_action_estimate -
                                                                  self.S_A_values[state][action])

        self.history.append((state, action, reward))

class DynaQAgent(superAgent):

    def __init__(self, policy=choose_randomly, args={}, epsilon = 0.1, alpha = 0.15, patience=None, planning_steps=20):
        super().__init__()
        self.name = "DynaQ-Learning Agent"
        self.set_policy(policy, args)
        self.learning_rate = alpha
        self.patience=patience

        self.planning_steps = planning_steps

        self.transition_table = np.array([])
        self.reward_table = {}
        self.x_lookup = {}
        self.y_lookup = {}
        self.state_ix_lookup = {}
        self.state_list = []
        self.terminal_state = np.array([0,0,0,0])
        print("remember to set self.actions = env.action_space!")

    def initialize_state_actions(self, new_default=0, do_nothing_action=0, do_nothing_bonus=0):
        '''Extends the base Agent classes initialize_state_actions to include details on the transition and reward
        model used by the algorithm'''
        super().initialize_state_actions(new_default, do_nothing_action, do_nothing_bonus)

        all_states = self.discretizer.list_all_states()
        all_states.append(self.terminal_state)
        all_states = [tuple(s) for s in all_states]
        self.state_list = all_states

        all_actions = np.arange(self.actions.n)

        all_sa = [(s, a) for a in all_actions for s in all_states]

        self.state_action_list = all_sa
        self.x_lookup = {(state, action): i for i, (state,action) in enumerate(all_sa)}
        self.y_lookup = {state: i for i, state in enumerate(all_states)}
        self.state_ix_lookup = {self.y_lookup[k]: k for k in self.y_lookup}
        self.sa_ix_lookup = {self.x_lookup[k]: k for k in self.x_lookup}
        self.transition_table = np.zeros((len(self.x_lookup), len(self.y_lookup)))
        self.reward_table = {}

    def get_action(self, state, actions, period):
        # This manages all aspects of the Q-learning algorithm including bootstrapping of the reward.
        # First, the last step is handled.
        # Then, the next action is selected.

        action = super()._follow_policy(state, actions, period)
        return action

    def end_episode(self, reward):
        # reset history
        self.history = []

        # Check if the policy has converged if a patience has been set
        if self.patience is not None:
            super().check_policy_convergence()

        self.past_S_A_values = deepcopy(self.S_A_values.copy())

    def observe_sars(self, state, action, reward, next_state):
        if (next_state == self.terminal_state).all():
            greedy_action_estimate = 0
            next_state = tuple(self.terminal_state)
        else:
            next_state = tuple(self.discretize_space(next_state))
            greedy_action = _get_max_dict_val(self.S_A_values[next_state], self.default_SA_estimate)
            greedy_action_estimate = self.S_A_values[next_state][greedy_action]

        state = tuple(self.discretize_space(state))
        self.S_A_values[state][action] += self.learning_rate*(reward + greedy_action_estimate -
                                                                  self.S_A_values[state][action])

        self.history.append((state, action, reward))

        # Update the Model
        # Update the state-action-state transition
        self.transition_table[self.x_lookup[(state, action)], self.y_lookup[next_state]] += 1

        try:
            self.reward_table[(state, action, next_state)] += \
                self.learning_rate * (reward - self.reward_table[state, action, next_state])
        except KeyError:
            self.reward_table[(state, action, next_state)] = reward

        for sweeps in range(self.planning_steps):
            state = random.choice(list(self.S_A_values.keys()))
            action = random.choice(range(self.actions.n))

            # Get next state predicted by the model:
            p_sas = self.transition_table[self.x_lookup[(state, action)], :]

            if np.sum(p_sas) == 0:
                continue

            # Choose randomly from the next states based on their probability of being visited but ignoring terminal
            p_sas = p_sas / np.sum(p_sas)
            try:
                next_state = self.state_ix_lookup[np.random.choice(range(p_sas.size), p=p_sas)]
            except ValueError:
                print("Skipping a sweep because of p_sas error!")
                print(p_sas)
                print("sum of p_sas={}".format(np.sum(p_sas)))
                print(self.transition_table[self.x_lookup[(state, action)], :])
                continue
            if (next_state == self.terminal_state).all():
                greedy_action_estimate = 0
            else:
                greedy_action = _get_max_dict_val(self.S_A_values[next_state], self.default_SA_estimate)
                greedy_action_estimate = self.S_A_values[next_state][greedy_action]

            reward = self.reward_table[(state, action, next_state)]

            self.S_A_values[state][action] += self.learning_rate * (reward + greedy_action_estimate -
                                                                self.S_A_values[state][action])

class ThresholdAgent(superAgent):
    def __init__(self, threshold):
        super().__init__()
        self.set_policy(hard_limit, {'target_demand': threshold})
        self.name = 'ThresholdAgent'

    def get_action(self, state, actions, period):
        action = self._follow_policy(state, actions, period)
        return action




class MonteCarloAgent(superAgent):
    def __init__(self, policy=choose_randomly, args={}):
        super().__init__()
        self.set_policy(policy, args)
        self.name = 'MonteCarloAgent'
        self.subtype = 'on-policy'
        self.C_S_A = {}
        self.learning_rate = args.get('learning_rate', None)
        self.terminal_state = np.array([0,0,0,0])

    def get_action(self, state, actions, period):
        action = super()._follow_policy(state, actions, period)
        state = tuple(self.discretize_space(state))
        return action

    def initialize_state_actions(self, new_default=0, do_nothing_action=None, do_nothing_bonus=1):
        self.default_SA_estimate = new_default
        super().initialize_state_actions(new_default, do_nothing_action, do_nothing_bonus)

        all_states = self.discretizer.list_all_states()
        all_states.append(self.terminal_state)
        self.state_list = all_states

        all_actions = np.arange(self.actions.n)

        all_sa = [(s, a) for a in all_actions for s in all_states]

        for s in all_states:
            s = tuple(s)
            for a in all_actions:
                self.S_A_values.setdefault(s, {})[a] = new_default
                self.C_S_A.setdefault(s, {})[a] = 0
                self.S_A_frequency.setdefault(s, {})[a] = 0
                if do_nothing_action is not None and a == do_nothing_action:
                    self.S_A_values[s][a] += do_nothing_bonus # Break ties in favor of do nothing

    def observe_sars(self, state, action, reward, next_state):
        state = tuple(self.discretize_space(state))
        self.history.append((state, action, reward))
        self.S_A_frequency[state][action] += 1

    def end_episode(self, reward):
        if self.subtype == 'on-policy':
            for s, a, r in list(set(self.history)):
                past_frequency =  self.S_A_frequency.setdefault(s, {}).get(a, 0)
                past_value = self.S_A_values.setdefault(s, {}).get(a, reward)
                if self.learning_rate is None:
                    self.S_A_values[s][a] = \
                        (self.S_A_values[s].get(a, 0)*past_frequency + reward) / (past_frequency + 1)
                else:
                    self.S_A_values[s][a] = past_value + self.learning_rate * (reward - past_value)

        elif self.subtype == 'off-policy':
            G = 0 # assumes all reward at end of episode
            W = 1
            for h in self.history[::-1]:
                s = h[0]
                a = h[1]
                r = h[2]

                G += r

                past_C = self.C_S_A.setdefault(s, {}).get(a, 0)
                self.C_S_A[s][a] += W
                past_value = self.S_A_values.setdefault(s, {}).get(a, self.default_SA_estimate)

                self.S_A_values[s][a] = past_value + W / self.C_S_A[s][a] * (G - past_value)

                if max(self.S_A_values[s].values()) != self.S_A_values[s][a]:
                    print("breaking. Action {} not greedy at state {} given reward {}, S_A_values of {}".format(a, s, G, self.S_A_values[s]))
                    break
                else:
                    # We know we are taking the greedy action. so, we also know that probability b must be
                    # 1-eta
                    # We also know we are trying to determine the optimal policy so pi(At_St) = 1.
                    eta = self.policy_args['eta']
                    b_S_A = 1-eta
                    W *= 1 / b_S_A

        else:
            raise NotImplementedError("Only on-policy and off-policy subtypes are implemented.")

        if self.patience is not None:
            super().check_policy_convergence()

        self.past_S_A_values = deepcopy(self.S_A_values)

        self.history = []

