import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize


class OneClassDataset(Dataset):
    def __init__(self, dataset: Dataset, one_class_labels=[], zero_class_labels=[], augmentation=True):
        self.dataset = dataset
        self.one_class_labels = one_class_labels
        self.filtered_indexes = []

        valid_labels = one_class_labels + zero_class_labels
        for i, (x, l) in enumerate(self.dataset):
            if l in valid_labels:
                self.filtered_indexes.append(i)

        # transform = Compose([ToTensor(), Resize((32, 32)), Normalize(mean=(0.5), std=(0.5))])
        # to_tensor = ToTensor()
        # self.xs = []
        # self.ls = []
        # for findex in self.filtered_indexes:
        #     x, l = self.dataset[findex]
        #     self.xs.append(to_tensor(x))
        #     self.ls.append(l)
        #
        # self.xs = torch.stack(self.xs)
        # self.ls = torch.tensor(self.ls)

        self.augmentation = augmentation

        # self.aug_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(32),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #     transforms.RandomGrayscale(p=0.2)])

        self.toTensor = ToTensor()
        # self.norm_transform = Compose([Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]
        l = 1 if l in self.one_class_labels else 0

        # if self.augmentation:
        #     return self.norm_transform(self.aug_transform(x)), l
        # else:
        #     return self.norm_transform(x), l

        return self.toTensor(x), l

    def __len__(self):
        return len(self.filtered_indexes)