'''
Created on 20 fevrier 2026
'''

#--------------------------------------------------------------------#
#                                                                    #
#                          external imports                          #
#                                                                    #
#--------------------------------------------------------------------#

from math import sqrt, log
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



class LinUCB1():

    def __init__(self, arms=None, dimension_context=None): 
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "LinUCB1"

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }
        
        self.arm_chosen = None
        # threshold used to compute rewards, actual feedback is compared to it
        # Follow the simulator metric, but this can be changed.
        self.threshold = 4

        self.dimension_context = dimension_context

        self.alpha = 1 

        self.A = np.array([np.identity(self.dimension_context) for i in range(len(self.ground_arms))]) # A is a list of matrices, one for each arm
        self.b = np.array([np.zeros(self.dimension_context) for i in range(len(self.ground_arms))]) # b is a list of vectors, one for each arm
        
        
        # -------------------------------------------------------------------

    def run(self, observed_value, user_context=None):
        self.current_context = user_context

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
        arm_pool_size = len(self.arms_pool['arm_id'])
        expected_payoff = np.zeros(arm_pool_size) - 1
        x=self.current_context
        i=0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm][0]
            A_inv = np.linalg.inv(self.A[arm_pos])
            theta = np.dot(A_inv, self.b[arm_pos])

            expected_payoff[i] = np.dot(theta, x) + self.alpha * sqrt(np.dot(x, np.dot(A_inv, x)))


            i += 1 
        arm_chosen_index = np.argmax(expected_payoff) 
        
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
        x=self.current_context
        self.A[arm_pos] += np.outer(x, x)
        self.b[arm_pos] += observed_reward * x
        self.arms_payoff_vectors["cumulated_rewards"][self.arm_chosen] += observed_reward
        self.arms_payoff_vectors["tries"][self.arm_chosen] += 1
                  
        
        # -------------------------------------------------------------------

    # =======================================================================
