from sys import maxsize

from torch.distributions import MultivariateNormal


class training_result:
    def __init__(self, config):
        self.config = config

        self.min_condition_no_model = None
        self.min_condition_no = maxsize
        self.min_condition_no_distribution = None

    def model_state_dict(model):
        state_dict = model.state_dict().copy()
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu()
        return state_dict

    def update(self, model, mean, cov, condition_no):

        if condition_no > 0 and condition_no < self.min_condition_no:
            self.min_condition_no = condition_no
            self.min_condition_no_model = training_result.model_state_dict(model)
            self.min_condition_no_distribution = MultivariateNormal(mean, cov)

        self.latest_model = training_result.model_state_dict(model)
        self.latest_distribution = MultivariateNormal(mean, cov)
