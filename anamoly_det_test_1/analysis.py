from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from torch import det
from torch.distributions import MultivariateNormal
from torch.nn import CosineSimilarity
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from anamoly_det_test_1.datasets import OneClassDataset
from anamoly_det_test_1.models import Encoder


def analyse(config):

    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])
    dataset = CIFAR10(root='../', train=True, download=True)
    inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier)
    outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier)
    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    train_dataset = train_inlier_dataset
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    prob_sum = None
    prob_count = 0
    for i, result_file in enumerate(listdir(config['result_folder'])):

        with open(config['result_folder'] + result_file, 'rb') as file:
            result = load(file)

        validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        model = Encoder(config)
        model.load_state_dict(result.latest_model)
        model.eval()
        model = model.cuda()

        distribution = result.latest_distribution

        prob = []
        labels = []
        with torch.no_grad():
            for x, l in validation_dataloader:
                x = x.cuda()
                x = model(x)
                log_prob = np.float128( distribution.log_prob(x).cpu().numpy())
                prob.append(np.exp(log_prob))
                labels.append(l)

        prob = np.concatenate(prob)
        labels = torch.cat(labels)

        cov = distribution.covariance_matrix
        d = cov.shape[0]
        eig_val = torch.real(torch.linalg.eig(cov)[0]).to(torch.float64)
        Z = np.sqrt((np.power(np.float128(2 * np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()))))
        prob = prob * Z
        prob = torch.tensor(np.float64(prob))

        if torch.any(torch.isnan(prob)).item():
            print('prob is nan')
            continue

        assert torch.max(prob).item() <= 1, f'prob upper bound error, {torch.max(prob).item()}'
        assert torch.min(prob).item() >= 0, 'prob lower bound error'

        if prob_sum == None:
            prob_sum = prob
        else:
            prob_sum += prob

        prob_count += 1

        # if (i+1) % 10 == 0:
        print(f'{result_file}, roc_score: {roc_auc_score(labels.cpu().numpy(), (prob_sum / prob_count).cpu().numpy())}, roc: {roc_auc_score(labels.cpu().numpy(), prob.cpu().numpy())}')

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count

    assert torch.max(prob).item() <= 1, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'

    scores = prob.cpu().numpy()
    targets = labels.cpu().numpy()

    roc = roc_auc_score(targets, scores)
    print(f'class: {config["class"]}, roc_score: {roc}, not_nan: {prob_count}')

    return roc
