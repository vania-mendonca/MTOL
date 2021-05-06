

class TaskModel:

    def __init__(self, model_name, predictions, humanDAs, bleu_scores, comet_scores, num_models):
        self._model_name = model_name
        self._cumulative_reward = 0
        self._avg_reward = 0
        self._weight = 1
        self._weight_as_probability = 1 / num_models
        self._qvalue = 0
        self._model_predictions = predictions
        self._model_humanDAs = humanDAs
        self._model_bleu_scores = bleu_scores
        self._model_comet_scores = comet_scores

    @property
    def model_name(self):
        return self._model_name

    @property
    def avg_reward(self):
        return self._avg_reward

    @avg_reward.setter
    def avg_reward(self, ar):
        self._avg_reward = ar

    @property
    def cumulative_reward(self):
        return self._cumulative_reward

    @cumulative_reward.setter
    def cumulative_reward(self, cr):
        self._cumulative_reward = cr

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    @property
    def weight_as_probability(self):
        return self._weight_as_probability

    @weight_as_probability.setter
    def weight_as_probability(self, w):
        self._weight_as_probability = w

    @property
    def qvalue(self):
        return self._qvalue

    @qvalue.setter
    def qvalue(self, q):
        self._qvalue = q


    def get_human_DA(self, s_id):
        return self._model_humanDAs[s_id]


    def get_bleu_score(self, s_id):
        return self._model_bleu_scores[s_id]

    def get_comet_score(self, s_id):
        return self._model_comet_scores[s_id]


    def get_model_prediction(self, s_id):
        return self._model_predictions[s_id]


    def __str__(self):
        return self._model_name + " | Current weight: " + str(self.weight_as_probability)



