'''
Created on 23 mars 2024

@author: aletard
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

import random
import numpy as np


#--------------------------------------------------------------------#
#                                                                    #
#                          Packages import                           #
#                                                                    #
#--------------------------------------------------------------------#




#--------------------------------------------------------------------#
#                                                                    #
#                          Global Variables                          #
#                                                                    #
#--------------------------------------------------------------------#


#--------------------------------------------------------------------#
#                                                                    #
#                         Functions & Objects                        #
#                                                                    #
#--------------------------------------------------------------------#



class EGreedy():

    def __init__(self, arms=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "EGreedy"
        self.epsilon = 0.05

        # Vecteurs de récompenses cumulées et de nombre d'essais pour chaque bras
        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        self.threshold = 4
        
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

        arm_chosen_index = -1
        # Algorithm Initialization
        if np.min(self.arms_payoff_vectors["tries"]) == 0 :
            i=0
            for arm in self.arms_pool['arm_id']:
                # Get index of the arm in numpies for payoff and tries
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]
                
                if (self.arms_payoff_vectors["tries"][arm_pos] < 1) :
                    arm_chosen_index = i
                    break
                i += 1

        if arm_chosen_index == -1 : 

            # Random exploration
            n = random.uniform(0., 1.) # Tirage d'un nombre aléatoire entre 0 et 1
            if n < self.epsilon: # Si le nombre tiré est inférieur à epsilon, on choisit un bras aléatoirement
                    arm_chosen_index = random.choice(self.arms_pool.index) # Choix aléatoire parmi les bras disponibles dans la pool


            # Exploitation                
            else :
                arm_pool_size = len(self.arms_pool['arm_id']) # Taille de la pool d'actions disponibles
                expected_payoff = np.zeros(arm_pool_size) - 1 # Initialisation du tableau des payoffs attendus pour chaque bras de la pool
                i=0 # Index pour parcourir la pool d'actions
                for arm in self.arms_pool['arm_id']: # Parcours de chaque bras de la pool d'actions
                    arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0] # Récupération de la position du bras dans les tableaux de récompenses et d'essais
                    # Calcul du payoff attendu pour le bras actuel : récompense cumulée divisée par le nombre d'essais
                    expected_payoff[i] = self.arms_payoff_vectors["cumulated_rewards"][arm_pos] / self.arms_payoff_vectors["tries"][arm_pos]
 
                    i += 1 # Incrémentation de l'index pour le prochain bras de la pool
                arm_chosen_index = np.argmax(expected_payoff) # Choix de l'index du bras qui a le payoff attendu le plus élevé
                # arm_chosen_index correspond à l'index dans la pool d'actions, pas dans les tableaux de récompenses et d'essais
        
        arm_chosen = self.arms_pool["arm_id"][arm_chosen_index] # Récupération de l'identifiant du bras choisi à partir de la pool d'actions en utilisant l'index sélectionné
            
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
        self.arms_payoff_vectors["cumulated_rewards"][self.arm_chosen] += observed_reward
        self.arms_payoff_vectors["tries"][self.arm_chosen] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================
