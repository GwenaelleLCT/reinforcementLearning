'''
Created on 24 mars 2026
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

import numpy as np


#--------------------------------------------------------------------#
#                                                                    #
#                        Functions & Objects                         #
#                                                                    #
#--------------------------------------------------------------------#


class CTS():

    def __init__(self, arms=None, dimension_context=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "CTS"

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4

        # Paramètre de variance (v) pour l'exploration
        # Comme pour alpha dans LinUCB, on peut le baisser (ex: 0.05) pour maximiser l'exploitation
        self.v = 0.01

        self.dimension_context = dimension_context +1 # Adding 1 for the bias term

        # Initialisation des matrices A et vecteurs b (identique à LinUCB)
        self.A = np.array([np.identity(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        self.b = np.array([np.zeros(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        
        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):
        
        # Ajout du terme de biais (1.0) pour des performances optimales
        self.current_context = np.append(user_context, 1.0)

        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen

        # -------------------------------------------------------------------

    def init_choice(self, observation):

        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)
        
        # -------------------------------------------------------------------

    def choose_action(self):
        
        arm_pool_size = len(self.arms_pool['arm_id'])
        expected_payoffs = np.zeros(arm_pool_size) - 1

        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]

            # 1. Inversion de la matrice A
            A_inv = np.linalg.inv(self.A[arm_pos])
            
            # 2. Calcul du vecteur d'espérance moyen (theta hat)
            theta_hat = A_inv @ self.b[arm_pos]
            
            # 3. Échantillonnage de theta depuis une distribution normale multivariée
            # Moyenne = theta_hat
            # Matrice de covariance = v^2 * A_inv
            covariance_matrix = (self.v ** 2) * A_inv
            
            # Tirage au sort du vecteur de paramètres
            sampled_theta = np.random.multivariate_normal(theta_hat, covariance_matrix)
            
            # 4. Calcul du score avec le theta tiré au sort
            x = self.current_context
            expected_payoffs[i] = sampled_theta @ x
            
            i += 1
            
        arm_chosen_index = np.argmax(expected_payoffs) 
        arm_chosen = self.arms_pool["arm_id"][arm_chosen_index]
            
        return arm_chosen

        # -------------------------------------------------------------------

    def evaluate(self, observation):

        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1

        return reward

        # -------------------------------------------------------------------

    def update(self, observation):

        observed_reward = self.evaluate(observation)
        
        arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == self.arm_chosen][0]
        x = self.current_context
        
        # Mise à jour de A et b (identique à LinUCB)
        self.A[arm_pos] += np.outer(x, x)
        self.b[arm_pos] += observed_reward * x
        
        self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += observed_reward
        self.arms_payoff_vectors["tries"][arm_pos] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================