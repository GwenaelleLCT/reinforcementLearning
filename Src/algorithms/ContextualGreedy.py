'''
Created on 24 mars 2026
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

import numpy as np
import random


#--------------------------------------------------------------------#
#                                                                    #
#                        Functions & Objects                         #
#                                                                    #
#--------------------------------------------------------------------#


class ContextualGreedy():

    def __init__(self, arms=None, dimension_context=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "ContextualGreedy"
        self.epsilon = 0.05 

        # Vecteurs de récompenses cumulées et de nombre d'essais pour chaque bras
        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4

    # variables necessaires pour le traitement du contexte et la mise à jour des paramètres
        # dimension du contexte
        self.dimension_context = dimension_context

        # Initialisation des matrices A et vecteurs b
        self.A = np.array([np.identity(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        self.b = np.array([np.zeros(self.dimension_context) for arm in range(len(self.ground_arms))]) 
        
        
    def run(self, observed_value, user_context=None):
        
        # Stockage du contexte actuel pour le choix d'action et la mise à jour
        self.current_context = np.append(user_context)

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
        
        # Random exploration
        n = random.uniform(0., 1.) # Tirage d'un nombre aléatoire entre 0 et 1
        if n < self.epsilon: # Si le nombre tiré est inférieur à epsilon, on choisit un bras aléatoirement
                arm_chosen_index = random.choice(self.arms_pool.index) # Choix aléatoire parmi les bras disponibles dans la pool
            
        # Exploitation                
        else :
            arm_pool_size = len(self.arms_pool['arm_id']) # Taille de la pool d'actions disponibles
            expected_payoffs = np.zeros(arm_pool_size) - 1 # Initialisation du tableau des payoffs attendus pour chaque bras de la pool
            i = 0 # Index pour parcourir la pool d'actions
            for arm in self.arms_pool['arm_id']: # Parcours de chaque bras de la pool d'actions
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0] # Récupération de la position du bras dans les tableaux de récompenses et d'essais
                # Inversion de la matrice A et calcul de theta pour le bras actuel
                A_inv = np.linalg.inv(self.A[arm_pos]) 
                theta = A_inv @ self.b[arm_pos]
                
                # Calcul de la récompense attendue (theta^T * x)
                x = self.current_context # Contexte actuel de l'utilisateur
                expected_payoffs[i] = theta @ x # Produit scalaire entre theta et le contexte pour obtenir la récompense attendue pour ce bras
                
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
        
        # Mise à jour de A et b
        self.A[arm_pos] += np.outer(x, x)
        self.b[arm_pos] += observed_reward * x
        
        self.arms_payoff_vectors["cumulated_rewards"][arm_pos] += observed_reward
        self.arms_payoff_vectors["tries"][arm_pos] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================