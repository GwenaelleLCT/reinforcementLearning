'''
LinUCB1 (hérite de ContextualAlgorithm)
'''
from cmath import sqrt

from Src.algorithms.ContextualAlgorithms import ContextualAlgorithms
import numpy as np
import random

class LinUCB12(ContextualAlgorithms):

    def __init__(self, arms=None, dimension_context=None): 
        # Appelle du __init__ de la classe mère
        super().__init__(arms, dimension_context, name="LinUCB12")
        self.alpha= 1

    # Méthode choose_action
    def choose_action(self):
        arm_pool_size = len(self.arms_pool['arm_id'])
        expected_payoff = np.zeros(arm_pool_size) - 1
        x = self.current_context
        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]

            A_inv = np.linalg.inv(self.A[arm_pos])
            theta = np.dot(A_inv, self.b[arm_pos])

            expected_payoff[i] = np.dot(theta, x) + self.alpha * sqrt(np.dot(x, np.dot(A_inv, x)))

            i += 1
        arm_chosen_index = np.argmax(expected_payoff) 

        return self.arms_pool["arm_id"][arm_chosen_index]