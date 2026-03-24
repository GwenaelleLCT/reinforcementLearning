'''
CTS (hérite de ContextualAlgorithm)
'''
from Src.algorithms.ContextualAlgorithms import ContextualAlgorithms
import numpy as np
import random

class CTS2(ContextualAlgorithms):

    def __init__(self, arms=None, dimension_context=None): 
        # Appelle du __init__ de la classe mère
        super().__init__(arms, dimension_context, name="CTS2")
        self.v= 0.01

    # Méthode choose_action
    def choose_action(self):
        arm_pool_size = len(self.arms_pool['arm_id'])
        sampled_values = np.zeros(arm_pool_size)

        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]

            A_inv = np.linalg.inv(self.A[arm_pos])
            theta_hat = A_inv @ self.b[arm_pos]

            covariance_matrix = (self.v**2) * A_inv
            sampled_theta = np.random.multivariate_normal(theta_hat, covariance_matrix)
            
            x = self.current_context
            sampled_values[i] = sampled_theta @ x
            i += 1
        arm_chosen_index = np.argmax(sampled_values) 

        return self.arms_pool["arm_id"][arm_chosen_index]