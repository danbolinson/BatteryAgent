from .policies import hard_limit, greedy_epsilon, choose_randomly
from .policies import _get_max_dict_val
import operator
from copy import deepcopy

from gym.spaces.box import Box, Space

import numpy as np

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
    def __init__(self):
        self.policy = None
        self.name = 'superclass'
        self.S_A_values = {}
        self.S_A_frequency = {}
        self.history = []
        self.default_SA_estimate = 0
        self.patience = None
        self.patience_counter = 0
        self.past_S_A_values = deepcopy(self.S_A_values)
        self.discretizer = Discretizer()
        self.discrete = False
        self.actions = Space()

    def set_discretizer(self, discretizer):
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

    def get_action(self, state, actions, period):
        # This should probably be overridden by whatever the agent needs to do.
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

    def set_greedy_policy(self, eta=0.2):
        self.historical_S_A_values = self.S_A_values.copy()
        args = {
            'eta': eta,
            'state_action_values': self.historical_S_A_values
        }

        self.set_policy(greedy_epsilon, args)
        self.S_A_frequency = {}

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

    def initialize_state_actions(self, default=0, do_nothing_action=0, do_nothing_bonus=0):
        '''Initializes the S_A Tables for values and frequencies based on the states and actions known to the agent.'''
        if not isinstance(self.discretizer, Box_Discretizer):
            raise NotImplementedError("initializing state space is only set up for 1D Box state space")

        states = self.discretizer.list_all_states()
        np_actions = np.arange(self.actions.n)

        for s in states:
            s = tuple(s)
            for a in np_actions:
                self.S_A_values.setdefault(s, {})[a] = default
                self.S_A_frequency.setdefault(s, {})[a] = 0

                if do_nothing_action is not None and a == do_nothing_action:
                    self.S_A_values[s][a] += do_nothing_bonus  # Break ties in favor of do nothing

class PrioritizedSweepingAgent(superAgent):

    def __init__(self, policy=choose_randomly, args={}, learning_rate = 0.1, patience=None, n_sweeps=5, sensitivity = 100):
        super().__init__()
        self.name = "Prioritized Sweeping Agent"
        self.set_policy(policy, args)
        self.learning_rate = learning_rate
        self.patience = None
        self.P_queue = []
        self.n_sweeps = n_sweeps
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
        next_state = tuple(self.discretize_space(next_state))

        # Update the state-action-state transition
        self.transition_table[self.x_lookup[(state, action)], self.y_lookup[next_state]] += 1
        if self.reward_table[self.y_lookup[state]] == 0:
            self.reward_table[self.y_lookup[state]] = reward
        else:
            self.reward_table[self.y_lookup[state]] += \
                self.learning_rate * (reward - self.reward_table[self.y_lookup[state]])

        # Get the change and priority of the change
        greedy_next_action = _get_max_dict_val(self.S_A_values[next_state])
        P = abs(reward + self.S_A_values[next_state][greedy_next_action] - self.S_A_values[state][action])
        insert_p(P)

        # Do the planning step by emptying P_queue
        for i in range(self.n_sweeps):
            if self.P_queue == []:
                break

            (state, action), _ = self.P_queue.pop(0)
            reward = self.reward_table[self.y_lookup[state]]

            p_sas = self.transition_table[self.x_lookup[(state, action)], :]
            # Choose randomly from the past states based on their probability of being visited
            p_sas = p_sas / np.sum(p_sas)
            next_state = self.state_list[np.random.choice(np.arange(len(self.state_list)), p=p_sas)]

            greedy_next_action = _get_max_dict_val(self.S_A_values[next_state])

            self.S_A_values[state][action] += \
                                        self.learning_rate * (reward + self.S_A_values[next_state][greedy_next_action])

            lead_to_s = state
            greedy_next_action = _get_max_dict_val(self.S_A_values[lead_to_s])

            for ix in np.nditer(np.nonzero(self.transition_table[:, self.y_lookup[lead_to_s]])):
                sa_cnt = self.transition_table[ix, self.y_lookup[lead_to_s]]
                (state, action) = self.sa_ix_lookup[int(ix)]
                reward = self.reward_table[self.y_lookup[state]]
                P = abs(reward + self.S_A_values[lead_to_s][greedy_next_action] - self.S_A_values[state][action])
                insert_p(P)

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

        self.Q_learning(state)
        action = super()._follow_policy(state, actions, period)
        return action

    def Q_learning(self, state=None):
        if len(self.history) == 0:
            pass
        else:
            last_state, last_action, reward = self.history[-1]
            prev_estimate = self.S_A_values.setdefault(last_state, {}).get(last_action, self.default_SA_estimate)
            if state is None:
                # implies end of episode.
                greedy_action = 0
                greedy_action_estimate = 0
            else:
                greedy_action = _get_max_dict_val(self.S_A_values[state], self.default_SA_estimate)
                greedy_action_estimate = self.S_A_values[state][greedy_action]

            self.S_A_values[last_state][last_action] = prev_estimate + \
                                                       self.learning_rate * (
                                                       reward + greedy_action_estimate - prev_estimate)


    def collect_reward(self, reward, state, action):
        self.history.append((state, action, reward))

    def end_episode(self, reward):
        #add the final episodic reward to the end of the history
        self.history[-1] = (self.history[-1][0], self.history[-1][1], self.history[-1][2] + reward)

        # run q-learning and reset history
        self.Q_learning()
        self.history = []

        # Check if the policy has converged if a patience has been set
        if self.patience is not None:
            super().check_policy_convergence()

        self.past_S_A_values = deepcopy(self.S_A_values.copy())


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

    def get_action(self, state, actions, period):
        action = super()._follow_policy(state, actions, period)
        state = tuple(self.discretize_space(state))
        self.history.append((state, action))
        return action

    def initialize_state_actions(self, states, actions, default=0, do_nothing_action=None, do_nothing_bonus=1):
        for s in states:
            s = tuple(s)
            for a in actions:
                self.S_A_values.setdefault(s, {})[a] = default
                self.C_S_A.setdefault(s, {})[a] = 0
                self.S_A_frequency.setdefault(s, {})[a] = 0
                if do_nothing_action is not None and a == do_nothing_action:
                    self.S_A_values[s][a] += do_nothing_bonus # Break ties in favor of do nothing

    def end_episode(self, reward):
        if self.subtype == 'on-policy':
            for s, a in list(set(self.history)):
                past_frequency =  self.S_A_frequency.setdefault(s, {}).get(a, 0)
                past_value = self.S_A_values.setdefault(s, {}).get(a, reward)
                if self.learning_rate is None:
                    self.S_A_values[s][a] = \
                        (self.S_A_values[s].get(a, 0)*past_frequency + reward) / (past_frequency + 1)
                else:
                    self.S_A_values[s][a] = past_value + self.learning_rate * (reward - past_value)

                self.S_A_frequency.setdefault(s, {})[a] = past_frequency + 1
        elif self.subtype == 'off-policy':
            G = reward # assumes all reward at end of episode
            W = 1
            for h in self.history[::-1]:
                s = h[0]
                a = h[1]

                past_C = self.C_S_A.setdefault(s, {}).get(a, 0)
                past_value = self.S_A_values.setdefault(s, {}).get(a, 0)
                self.C_S_A[s][a] += W
                self.S_A_values[s][a] = past_value + W / self.C_S_A[s][a] * (G - past_value)

                if _get_max_dict_val(self.S_A_values[s]) != a:
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

