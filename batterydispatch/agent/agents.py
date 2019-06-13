from .policies import hard_limit, greedy_epsilon, choose_randomly
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

    def set_policy(self, policy, policy_args={}):
        self.policy = policy
        self.policy_args = policy_args

    def end_episode(self, reward=0):
        pass


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
                self.S_A_values.set_default(s, {})[a] = default
                self.S_A_frequency.set_default(s, {})[a] = 0

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

    def set_greedy_policy(self, learning_rate=0.2):
        self.historical_S_A_values = self.S_A_values.copy()
        args = {
            'learning_rate': learning_rate,
            'state_action_values': self.historical_S_A_values
        }

        self.set_policy(greedy_epsilon, args)
        self.S_A_frequency = {}

