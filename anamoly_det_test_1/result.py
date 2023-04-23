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

    def set_aug_transform(self, aug_transforms):
        self.aug_transforms = aug_transforms

    def update(self, models, means, covs, condition_no):

        # if condition_no > 0 and condition_no < self.min_condition_no:
        #     self.min_condition_no = condition_no
        #     self.min_condition_no_model = training_result.model_state_dict(model)
        #     self.min_condition_no_distribution = MultivariateNormal(mean, cov)

        self.latest_models = []
        self.latest_distributions = []
        for i in range(self.config['encoders_n']):
            self.latest_models.append(training_result.model_state_dict(models[i]))
            self.latest_distributions.append(MultivariateNormal(means[i], covs[i]))
