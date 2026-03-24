'''
Contextual Greedy (hérite de ContextualAlgorithm)
'''
from Src.algorithms.ContextualAlgorithms import ContextualAlgorithms
import numpy as np
import random

class ContextualGreedy2(ContextualAlgorithms):

    def __init__(self, arms=None, dimension_context=None): 
        # Appelle du __init__ de la classe mère
        super().__init__(arms, dimension_context, name="ContextualGreedy2")
        self.epsilon = 0.05 

    # Méthode choose_action
    def choose_action(self):
        
        # Random exploration
        n = random.uniform(0., 1.) # Tirage d'un nombre aléatoire entre 0 et 1
        if n < self.epsilon: # Si le nombre tiré est inférieur à epsilon, on choisit un bras aléatoirement
                arm_chosen_index = random.choice(self.arms_pool.index) # Choix aléatoire parmi les bras disponibles dans la pool
            
        # Exploitation                
        else :
            arm_pool_size = len(self.arms_pool['arm_id'])
            expected_payoffs = np.zeros(arm_pool_size) - 1

            i = 0
            for arm in self.arms_pool['arm_id']:
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]

                A_inv = np.linalg.inv(self.A[arm_pos])
                theta = A_inv @ self.b[arm_pos]
                
                x = self.current_context
                expected_payoffs[i] = theta @ x
                i += 1
            arm_chosen_index = np.argmax(expected_payoffs) 

        return self.arms_pool["arm_id"][arm_chosen_index]