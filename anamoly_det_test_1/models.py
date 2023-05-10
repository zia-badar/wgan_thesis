from pickle import load

import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d, ReLU, BatchNorm1d, LeakyReLU


class AbsActivation(nn.Module):

    def __init__(self):
        super(AbsActivation, self).__init__()

        self.base = 0.001
        self.slope = 0.001

    def forward(self, x):
        ret = self.base + torch.abs(x) * self.slope
        return ret


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 2
        self.d = nn.Sequential(

            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            BatchNorm1d(num_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            BatchNorm1d(num_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=1)
        )

    def forward(self, x):
        # return torch.sum(self.d(x), dim=-1)
        return torch.squeeze(self.d(x))


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.e = nn.Sequential(

            # nn.Flatten(),
            # nn.Linear(in_features=config['data_dim'], out_features=(int)(config['data_dim']/2)),
            # nn.BatchNorm1d(num_features=(int)(config['data_dim']/2)),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=(int)(config['data_dim']/2), out_features=(int)(config['data_dim']/4)),
            # nn.BatchNorm1d(num_features=(int)(config['data_dim']/4)),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=(int)(config['data_dim']/4), out_features=(int)(config['data_dim']/8)),
            # nn.BatchNorm1d(num_features=(int)(config['data_dim']/8)),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=(int)(config['data_dim']/8), out_features=config['encoding_dim']),

            # Conv2d(in_channels=1, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=256),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=1024),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=1024, out_channels=config['encoding_dim'], kernel_size=4, bias=False),
        )

    def forward(self, x):
        return self.e(x).squeeze()


class Projection(nn.Module):

    def __init__(self, config):
        super(Projection, self).__init__()

        # self.projection_1 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.projection_2 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.translation_1 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))
        # self.translation_2 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))

        self.projection_1 = torch.tensor([[0.8560, 0.3906, 0.7770], [0.1772, 0.2052, 0.4125], [0.7921, 0.8567, 0.3301]])
        self.projection_2 = torch.tensor([[0.9458, 0.2717, 0.7411], [0.5602, 0.7715, 0.1062], [0.4664, 0.5055, 0.6179]])
        self.translation_1 = torch.tensor([-2.6553, 2.4304, 8.3437])
        self.translation_2 = torch.tensor([5.3460, 2.8367, 1.0990])
        #
        # with open('results/result_non_l', 'rb') as file:
        #     result = load(file)
        #
        # self.projection = result.projection.projection
        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=config['encoding_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(in_features=config['data_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

    def forward(self, x):
        projections = torch.empty((x.shape[0],) + self.projection_1.shape)
        translations = torch.empty((x.shape[0],) + self.translation_1.shape)
        seed = (torch.rand((x.shape[0],)) < 0.5)
        translations[seed] = self.translation_1
        translations[torch.logical_not(seed)] = self.translation_2
        projections[seed] = self.projection_1
        projections[torch.logical_not(seed)] = self.projection_2
        return torch.squeeze(x[:, None, :] @ projections) + translations
        #
        # with torch.no_grad():
        #     return self.projection(x)
