import numpy as np
from random import choices

class OnlineAlgorithm:
    
    def __init__(self, num_experts, decimal_places, eta, reward_function):
        self._reward_decimal_places = decimal_places
        self._eta = eta
        self._reward_function=reward_function
        self._chosen_expert = None
        self._weights = [1 for i in range(num_experts)]
        self._weights_as_probabilities = [1 / num_experts for i in range(num_experts)]
        
    @property
    def chosen_expert(self):
        return self._chosen_expert
    
    @property
    def weights_as_probabilities(self):
        return self._weights_as_probabilities

    
    def forecaster(self, experts, instance_id):
        self._chosen_expert = choices(range(0, len(experts)), self._weights_as_probabilities)[0]

        return experts[self._chosen_expert].get_model_prediction(instance_id)

    def update(self, experts, current_iteration, instance_id):
        pass


    def _reward_(self, expert, instance_id):

        if str.startswith(self._reward_function, "human"):
            reward_value = expert.get_human_DA(instance_id)

            if np.isnan(reward_value):
                if self._reward_function == "human": #human-zero
                    reward_value = 0
                elif self._reward_function == "human-avg":
                    reward_value = expert.avg_reward
                elif self._reward_function == "human-comet":                    
                    reward_value = expert.get_comet_score(instance_id)
            else:
                reward_value = reward_value * 0.01

        elif self._reward_function == "bleu":
            reward_value = expert.get_bleu_score(instance_id)
            reward_value = reward_value * 0.01
        elif self._reward_function == "comet":
            reward_value = expert.get_comet_score(instance_id)
        else:
            return # FIXME throw exception


        if not(self._reward_decimal_places == None):
            return np.round(reward_value, decimals=self._reward_decimal_places)
        else:
            return reward_value


################################################################################

class EWAF(OnlineAlgorithm):


    def update(self, experts, current_iteration, instance_id):
        num_experts = len(experts)

        self._weights = []
        self._weights_as_probabilities = []

        for expert in experts:

            print(expert.model_name)

            expert_reward = self._reward_(expert, instance_id)
            print("Current reward", expert_reward)

            expert.cumulative_reward = expert.cumulative_reward + expert_reward
            print("Total reward", expert.cumulative_reward)
            
            expert.avg_reward = expert.cumulative_reward / max(1, (current_iteration - 1))

            eta = np.sqrt((self._eta * np.log(num_experts)) / current_iteration)
            expert.weight = np.exp(eta * expert.cumulative_reward)
            self._weights.append(expert.weight)

        total_weight =  np.sum(self._weights)

        for expert in experts:
            expert.weight_as_probability = expert.weight / total_weight
            self._weights_as_probabilities.append(expert.weight_as_probability)


    def __str__(self):
        return "EWAF | decimal places=" + str(self._reward_decimal_places) + " | eta=" + str(self._eta) + " | reward=" + self._reward_function


################################################################################

class EXP3(OnlineAlgorithm):

    def update(self, arms, current_iteration, instance_id):

        num_arms = len(arms)

        arm_chosen = arms[self._chosen_expert]

        arm_reward = self._reward_(arm_chosen, instance_id)
        print("Current reward", arm_reward)

        arm_chosen.cumulative_reward = arm_chosen.cumulative_reward + (arm_reward / arm_chosen.weight_as_probability)
        
        arm_chosen.avg_reward = arm_chosen.cumulative_reward / max(1, (current_iteration - 1))

        print("Total reward", arm_chosen.cumulative_reward)

        eta = np.sqrt((2 * np.log(num_arms)) / (current_iteration * num_arms))
        arm_chosen.weight = np.exp(eta * arm_chosen.cumulative_reward)


        self._weights[self._chosen_expert] = arm_chosen.weight
        total_weight = np.sum(self._weights)

        arm_chosen.weight_as_probability = arm_chosen.weight / total_weight
        self._weights_as_probabilities = [ w / total_weight for w in self._weights]


    def __str__(self):
        return "EXP3 | decimal places=" + str(self._reward_decimal_places) + " | eta=" + str(self._eta) + " | reward=" + self._reward_function



################################################################################


def init_online_algorithm(algorithm, num_experts, decimal_places=None, eta_value=8,  reward_function="human"):

    if algorithm == "EWAF":
        return EWAF(num_experts, decimal_places, eta_value, reward_function)
    elif algorithm == "EXP3":
        return EXP3(num_experts, decimal_places, eta_value, reward_function)
    else:
        return # FIXME throw exception