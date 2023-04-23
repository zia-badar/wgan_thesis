import sys
from os import mkdir, rmdir, listdir
from pickle import dumps, dump
from time import localtime, mktime
import shutil

import numpy as np
import torch.nn
import torchvision.transforms.functional
from numpy import arange
from sklearn.metrics import roc_auc_score
from torch import softmax, sigmoid, nn, det
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, CosineSimilarity
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from tqdm import tqdm

from anamoly_det_test_1.DeterministicTransforms import ColorJitter, RandomResizedCrop, RandomHorizontalFlip

sys.path.append('/home/users/z/zia_badar/masterthesisgitlab/')

from anamoly_det_test_1.analysis import analyse
from anamoly_det_test_1.models import Discriminator, Encoder, Projection
from anamoly_det_test_1.datasets import OneClassDataset
from anamoly_det_test_1.result import training_result


def train_encoder(config):
    aug_transforms = []

    for _ in range(config['encoders_n']):
        aug_transforms.append(transforms.Compose(list(filter(lambda item: item is not None, [
            # transforms.RandomResizedCrop(32),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomGrayscale(p=0.2)

            RandomResizedCrop(32),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(0.4, 0.4, 0.4, 0.1) if torch.rand(1) < 0.8 else None,
            transforms.Grayscale(num_output_channels=3) if torch.rand(1) < 0.2 else None,
        ]))))

    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])
    dataset = CIFAR10(root='../', train=True, download=True)
    # for setting determinister parameters of transform
    for i in range(config['encoders_n']):
        _ = aug_transforms[i](dataset[0][0])

    inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier)
    outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier)
    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    train_dataset = train_inlier_dataset
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    fs = []
    es = []

    for _ in range(config['encoders_n']):
        fs.append(Discriminator(config).cuda())
        es.append(Encoder(config).cuda())

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    for i in range(config['encoders_n']):
        fs[i].apply(weights_init)
        es[i].apply(weights_init)

    def _next(iter):
        try:
            batch = next(iter)
            return False, batch
        except StopIteration:
            return True, None

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    encoder_dataloader_iter = iter( DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    optim_fs = []
    optim_es = []
    for i in range(config['encoders_n']):
        optim_fs.append(RMSprop(fs[i].parameters(), lr=config['lr'], weight_decay=config['weight_decay']))
        optim_es.append(RMSprop(es[i].parameters(), lr=config['lr'], weight_decay=config['weight_decay']))

    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    result = training_result(config)
    result_file_name = f'{config["result_folder"]}result_{(int)(mktime(localtime()))}'
    result.set_aug_transform(aug_transforms)
    norm_transform = Compose([Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    cosine_sim = CosineSimilarity(dim=-1)

    progress_bar = tqdm(range(1, config['encoder_iters']+1))

    for encoder_iter in progress_bar:

        for _ in range(config['discriminator_n']):
            empty, batch = _next(discriminator_dataloader_iter)
            if empty:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
                _, batch = _next(discriminator_dataloader_iter)

            x, _ = batch
            x = x.cuda()
            z = normal_dist.sample((x.shape[0], )).cuda()

            for i in range(config['encoders_n']):
                loss = -torch.mean(fs[i](z) - fs[i](es[i](norm_transform(aug_transforms[i](x)))))

                optim_fs[i].zero_grad()
                loss.backward()
                optim_fs[i].step()

                for parameter in fs[i].parameters():
                    parameter.data = parameter.data.clamp(-config['clip'], config['clip'])

        empty, batch = _next(encoder_dataloader_iter)
        if empty:
            encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
            _, batch = _next(encoder_dataloader_iter)

        x, _ = batch
        x = x.cuda()

        e_xs = []
        for i  in range(config['encoders_n']):
            e_xs.append(es[i](norm_transform(aug_transforms[i](x))))

        center = torch.mean(torch.nn.functional.normalize(torch.stack(e_xs), dim=-1), dim=0)
        center = center.detach()

        for i in range(config['encoders_n']):
            f_x = fs[i](e_xs[i])
            loss = -torch.mean(f_x + cosine_sim(center, e_xs[i]))

            optim_es[i].zero_grad()
            loss.backward()
            optim_es[i].step()

        progress_bar.set_description(f'loss: {loss.item()}')

        if encoder_iter % 1000 == 0:
            means = []
            covs = []
            for i in range(config['encoders_n']):
                mean, cov = evaluate_encoder(es[i], train_dataset, validation_dataset, config, norm_transform, aug_transforms[i])
                means.append(mean)
                covs.append(cov)

            result.update(es, means, covs, -1)

            # with open(result_file_name + f'_{encoder_iter}', 'wb') as file:
            #     dump(result, file)

    with open(result_file_name, 'wb') as file:
        dump(result, file)

def evaluate_encoder(encoder, train_dataset, validation_dataset, config, norm_transform, aug_transform):
    encoder.eval()

    with torch.no_grad():
        for dataset in [train_dataset]:
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            encodings = []
            for x, _ in train_dataloader:
                x = x.cuda()
                encodings.append(encoder(norm_transform(aug_transform(x))))

            encodings = torch.cat(encodings)
            mean = torch.mean(encodings, dim=0)
            cov = torch.cov(encodings.t(), correction=0)
            # eig_val, eig_vec = eig(cov)
            # condition_no = torch.max(eig_val.real) / torch.min(eig_val.real)

        # validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        # distribution = MultivariateNormal(mean, cov)
        #
        # prob = []
        # labels = []
        # with torch.no_grad():
        #     for x, l in validation_dataloader:
        #         x = x.cuda()
        #         labels.append(l)
        #         log_prob = np.float128(distribution.log_prob(encoder(x)).cpu().numpy())
        #         prob.append(np.exp(log_prob))
        #
        # prob = np.concatenate(prob)
        # labels = torch.cat(labels)
        #
        # cov = distribution.covariance_matrix
        # d = cov.shape[0]
        # eig_val = torch.real(torch.linalg.eig(cov)[0]).to(torch.float64)
        # Z = np.sqrt((np.power(np.float128(2 * np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()))))
        # prob = prob * Z
        #
        # print( f'roc: {roc_auc_score(labels.cpu().numpy(), prob)}')

    encoder.train()

    return mean, cov

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sum = 0
    for _class in range(10):
        # folder = list(filter(lambda f: f.endswith(str(_class)), listdir('/home/zia/Desktop/MasterThesis/anamoly_det_test_1/results/set_3/')))[0]
        config = {'batch_size': 64, 'epochs': 200, 'encoding_dim': 32, 'encoder_iters': 1000, 'discriminator_n': 5, 'lr': 5e-5, 'weight_decay': 1e-6, 'clip': 1e-2, 'num_workers': 0, 'result_folder': f'results/set_{(int)(mktime(localtime()))}_{_class}/', 'encoders_n': 5 }
        config['lambda'] = 0

        config['class'] = _class
        mkdir(config['result_folder'])

        for _ in range(1):
            ret = False
            while ret == False:
                ret = train_encoder(config)
        roc = analyse(config)
        sum += roc
        print(f'class: {_class}, roc: {roc}')
        # shutil.rmtree(config['result_folder'])

    print(f'avg roc: {sum/10.}')
