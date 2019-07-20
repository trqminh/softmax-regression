import torch
from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        return x, y
