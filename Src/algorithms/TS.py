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


class TS():

    def __init__(self, arms=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "TS"

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4
        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen

        # -------------------------------------------------------------------

    def init_choice(self, observation):

        self.arm_chosen = -1
        # Ensuring algorithm only arms for which feedback have been provided by current user
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)
        
        # -------------------------------------------------------------------

    def choose_action(self):

        arm_pool_size = len(self.arms_pool['arm_id']) # Taille de la pool d'actions disponibles
        sampled_values = np.zeros(arm_pool_size) # Tableau pour stocker les valeurs échantillonnées pour chaque bras de la pool
        
        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0] # Récupération de la position du bras dans les tableaux de récompenses et d'essais
            
            # Calcul des Succès (S_i) et des Échecs (F_i)
            S_i = self.arms_payoff_vectors["cumulated_rewards"][arm_pos] # Nombre de succès pour le bras i
            F_i = self.arms_payoff_vectors["tries"][arm_pos] - S_i # Nombre d'échecs pour le bras i
            
            # Échantillonnage à partir de la distribution Beta (S_i + 1, F_i + 1)
            # Le "+1" sert d'initialisation (Prior de Bayes) quand le bras n'a jamais été joué
            sampled_values[i] = np.random.beta(S_i + 1, F_i + 1)
            
            i += 1

        # Choisir l'index du bras qui a obtenu la plus grande valeur lors du tirage
        arm_chosen_index = np.argmax(sampled_values)
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
        
        # Mise à jour des statistiques de base
        self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += observed_reward
        self.arms_payoff_vectors["tries"][arm_pos] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================