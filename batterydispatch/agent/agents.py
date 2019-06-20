from .policies import hard_limit, greedy_epsilon, choose_randomly
from .policies import _get_max_dict_val
import operator


class superAgent():
    def __init__(self):
        self.policy = None
        self.name = 'superclass'

    def follow_policy(self, state, actions, period):
        if self.policy is None:
            raise AssertionError("You must set a policy function before calling follow_policy.")

        return self.policy(state, actions, period, self.policy_args)

    def get_action(self, state, actions, period):
        # This should probably be overridden by whatever the agent needs to do.
        return self.follow_policy(state, actions, period)

    def collect_reward(self, reward, state, action=None):
        # This takes a reward for a specific state or state-action
        pass

    def set_policy(self, policy, policy_args={}):
        self.policy = policy
        self.policy_args = policy_args

    def end_episode(self, reward=0):
        pass


class QAgent(superAgent):

    def __init__(self, policy=choose_randomly, args={}, epsilon = 0.1, alpha = 0.15):
        super().__init__()
        self.name = "Q-Learning Agent"
        self.set_policy(policy, args)
        self.S_A_values = {}
        self.S_A_frequency = {}
        self.learning_rate = alpha
        self.history = []
        self.default_SA_estimate = 0

    def get_action(self, state, actions, period):
        # This manages all aspects of the Q-learning algorithm including bootstrapping of the reward.
        # First, the last step is handled.
        # Then, the next action is selected.

        self.Q_learning(state)
        action = super().follow_policy(state, actions, period)
        return action

    def Q_learning(self, state=None):
        if len(self.history) == 0:
            pass
        else:
            last_state, last_action, reward = self.history[-1]
            prev_estimate = self.S_A_values.setdefault(last_state, {}).get(last_action, self.default_SA_estimate)
            if state is None:
                # implies end of episode.
                greedy_action_estimate = 0
            else:
                greedy_action = _get_max_dict_val(self.S_A_values[state.as_tuple()], self.default_SA_estimate)
                greedy_action_estimate = self.S_A_values[state.as_tuple()][greedy_action]

            if greedy_action_estimate !=  prev_estimate and greedy_action != 0:
                print("WHAT THE FUCK")
                pass
            self.S_A_values[last_state][last_action] = prev_estimate + \
                                                       self.learning_rate * (
                                                       reward + greedy_action_estimate - prev_estimate)

    def initialize_state_actions(self, states, actions, default=0):
        for s in states:
            for a in actions:
                self.S_A_values.setdefault(s, {})[a] = default
                self.S_A_frequency.setdefault(s, {})[a] = 0
                if a == 0:
                    self.S_A_values[s][a] += 1 # Break ties in favor of do nothing

    def collect_reward(self, reward, state, action):
        self.history.append((state, action, reward))

    def end_episode(self, reward):
        self.history[-1] = (self.history[-1][0], self.history[-1][1], self.history[-1][2] + reward)
        self.Q_learning()
        self.history = []

    def set_greedy_policy(self, eta=0.05):
        self.historical_S_A_values = self.S_A_values.copy()
        args = {
            'eta': eta,
            'state_action_values': self.historical_S_A_values
        }

        self.set_policy(greedy_epsilon, args)
        self.S_A_frequency = {}


class ThresholdAgent(superAgent):
    def __init__(self, threshold):
        super().__init__()
        self.set_policy(hard_limit, {'target_demand': threshold})
        self.name = 'ThresholdAgent'

    def get_action(self, state, actions, period):
        action = self.follow_policy(state, actions, period)
        return action




class MonteCarloAgent(superAgent):
    def __init__(self, policy=choose_randomly, args={}):
        super().__init__()
        self.set_policy(policy, args)
        self.name = 'MonteCarloAgent'
        self.S_A_values = {}
        self.S_A_frequency = {}
        self.learning_rate = args.get('learning_rate', None)
        self.history = []

    def get_action(self, state, actions, period):
        action = super().follow_policy(state, actions, period)
        self.history.append((state.as_tuple(), action))
        return action

    def initialize_state_actions(self, states, actions, default=0):
        for s in states:
            for a in actions:
                self.S_A_values.setdefault(s, {})[a] = default
                self.S_A_frequency.setdefault(s, {})[a] = 0
                if a == 0:
                    self.S_A_values[s][a] += 1 # Break ties in favor of do nothing

    def end_episode(self, reward):
        for s, a in list(set(self.history)):
            past_frequency =  self.S_A_frequency.setdefault(s, {}).get(a, 0)
            past_value = self.S_A_values.setdefault(s, {}).get(a, reward)
            if self.learning_rate is None:
                self.S_A_values[s][a] = \
                    (self.S_A_values[s].get(a, 0)*past_frequency + reward) / (past_frequency + 1)
            else:
                self.S_A_values[s][a] = past_value + self.learning_rate * (reward - past_value)

            self.S_A_frequency.setdefault(s, {})[a] = past_frequency + 1
        self.history = []

    def set_greedy_policy(self, eta=0.2):
        self.historical_S_A_values = self.S_A_values.copy()
        args = {
            'eta': eta,
            'state_action_values': self.historical_S_A_values
        }

        self.set_policy(greedy_epsilon, args)
        self.S_A_frequency = {}

