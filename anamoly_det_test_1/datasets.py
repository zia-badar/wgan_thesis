import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


class OneClassDataset(Dataset):
    def __init__(self, dataset: Dataset, one_class_labels=[], zero_class_labels=[], transform=None, augmentation=False):
        self.dataset = dataset
        self.one_class_labels = one_class_labels
        self.transform = transform
        self.filtered_indexes = []

        valid_labels = one_class_labels + zero_class_labels
        for i, (x, l) in enumerate(self.dataset):
            if l in valid_labels:
                self.filtered_indexes.append(i)

        # transform = Compose([ToTensor(), Resize((32, 32)), Normalize(mean=(0.5), std=(0.5))])
        to_tensor = ToTensor()
        self.xs = []
        self.ls = []
        for findex in self.filtered_indexes:
            x, l = self.dataset[findex]
            self.xs.append(to_tensor(x))
            self.ls.append(l)

        self.xs = torch.stack(self.xs)
        self.ls = torch.tensor(self.ls)

        self.augmentation = augmentation

        self.aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])

        self.norm_transform = Compose([Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, item):
        # x, l = self.dataset[self.filtered_indexes[item]]
        #
        # if self.transform != None:
        #     x = self.transform(x)
        # l = 1 if l in self.one_class_labels else 0

        # return x, l
        x = self.xs[item]
        l = 1 if self.ls[item] in self.one_class_labels else 0


        if self.augmentation:
            return self.norm_transform(x), self.norm_transform(self.aug_transform(x)), l
        else:
            return self.norm_transform(x), l

    def __len__(self):
        return len(self.filtered_indexes)

class ProjectedDataset(Dataset):

    def __init__(self, train, distribution, projection):
        super(ProjectedDataset, self).__init__()

        projection.eval()
        with torch.no_grad():
            if train:
                self.dataset = projection(distribution.sample(sample_shape=(5000,)))
            else:
                self.dataset = projection(distribution.sample(sample_shape=(1500,)))

    def __getitem__(self, item):

        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
