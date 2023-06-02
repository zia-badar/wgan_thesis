from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from torch import det
from torch.distributions import MultivariateNormal
from torch.nn import CosineSimilarity
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize

from anamoly_det_test_1.datasets import OneClassDataset
from anamoly_det_test_1.models import Encoder


def analyse(config):
    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])
    cifar_train = CIFAR10(root='../', train=True, download=True)
    cifar_test = CIFAR10(root='../', train=False, download=True)
    train_dataset = OneClassDataset(cifar_train, one_class_labels=inlier)
    test_dataset = ConcatDataset([OneClassDataset(cifar_train, zero_class_labels=outlier), OneClassDataset(cifar_test, one_class_labels=inlier, zero_class_labels=outlier)])
    norm_transform = Compose([Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    cosine_sim = CosineSimilarity(dim=-1)
    prob_sum = None
    prob_count = 0
    roc_list = []
    roc_list2 = []
    for i, result_file in enumerate(listdir(config['result_folder'])):

        with open(config['result_folder'] + result_file, 'rb') as file:
            result = load(file)


        for encoder_n in range(config['encoders_n']):
            train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
            test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
            model = Encoder(config)
            model.load_state_dict(result.latest_models[encoder_n])
            model.eval()
            model = model.cuda()

            distribution = result.latest_distributions[encoder_n]

            prob = []
            labels = []
            cos_sim = []
            with torch.no_grad():
                encodings = []
                for x, _ in train_dataloader:
                    x = x.cuda()
                    encodings.append(model(norm_transform(result.aug_transforms[encoder_n](x))))
                encodings = torch.cat(encodings)

                # gamma = (10 / (torch.var(encodings).item() * encodings.shape[1]))
                # svm = OneClassSVM(kernel='rbf', gamma=gamma).fit(encodings.cpu().numpy())
                svm = OneClassSVM(kernel='linear').fit(encodings.cpu().numpy())

                val_x = []
                for x, l in test_dataloader:
                    x = x.cuda()
                    x = model(norm_transform(result.aug_transforms[encoder_n](x)))
                    val_x.append(x)
                    log_prob = np.float128( distribution.log_prob(x).cpu().numpy())
                    prob.append(np.exp(log_prob))
                    cos_sim.append(torch.max(cosine_sim(encodings, x.unsqueeze(1)), dim=1).values)
                    labels.append(l)

                val_x = torch.cat(val_x).cpu().numpy()


            cos_sim = torch.cat(cos_sim)
            prob = np.concatenate(prob)
            labels = torch.cat(labels)
            print(f'linear ocsvm roc: {roc_auc_score(labels, svm.score_samples(val_x))}')

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
                cos_sum = cos_sim
            else:
                prob_sum += prob
                cos_sum += cos_sim

            prob_count += 1

            # if (i+1) % 10 == 0:
            p = roc_auc_score(labels.cpu().numpy(), (prob_sum / prob_count).cpu().numpy())
            print(f'{result_file}, p(x|u,c), roc_score: {p}, roc: {roc_auc_score(labels.cpu().numpy(), prob.cpu().numpy())}')
            p2 = roc_auc_score(labels.cpu().numpy(), (cos_sum / prob_count).cpu().numpy())
            print(f'{result_file}, cos_sim, roc_score: {p2}, roc: {roc_auc_score(labels.cpu().numpy(), cos_sim.cpu().numpy())}')
            roc_list.append(p)
            roc_list2.append(p2)

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count
    cos_sim = cos_sum / prob_count

    assert torch.max(prob).item() <= 1, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'

    scores = prob.cpu().numpy()
    targets = labels.cpu().numpy()

    roc = roc_auc_score(targets, scores)
    roc2 = roc_auc_score(targets, cos_sim.cpu().numpy())
    print(f'class: {config["class"]}, p(x|u,c) roc_score: {roc}, cos_sim roc_score: {roc2}not_nan: {prob_count}')

    return roc, roc2, roc_list, roc_list2
