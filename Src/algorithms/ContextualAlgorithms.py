'''
Framework général pour les algorithmes contextuels
'''
import numpy as np

class ContextualAlgorithms():

    def __init__(self, arms=None, dimension_context=None, name="ContextualAlgorithms"): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = name

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4
        
        # dimension du contexte
        self.dimension_context = dimension_context

        # Matrices et vecteurs communs
        self.A = np.array([np.identity(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        self.b = np.array([np.zeros(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        
    def run(self, observed_value, user_context=None):
        self.current_context = user_context
        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        return self.arm_chosen

    def init_choice(self, observation):
        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    def evaluate(self, observation):
        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1
        return reward

    def update(self, observation):
        observed_reward = self.evaluate(observation)
        arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == self.arm_chosen][0]
        x = self.current_context
        
        self.A[arm_pos] += np.outer(x, x)
        self.b[arm_pos] += observed_reward * x
        
        self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += observed_reward
        self.arms_payoff_vectors["tries"][arm_pos] += 1

    def choose_action(self):
        # Méthode définie dans les algorithmes enfants
        pass