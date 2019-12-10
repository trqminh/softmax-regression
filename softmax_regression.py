import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class SoftmaxRegression:
    def __init__(self, dataset, num_class):
        self.data_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
        self.num_class = num_class
        self.weights = nn.Linear(dataset[0][0].shape[0], self.num_class, bias=True)
        self.model = nn.Sequential(self.weights, nn.Softmax())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        loss = 0.0
        for epoch in range(1000):

            for i, data in enumerate(self.data_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if epoch % 100 == 0:
                print('loss at {} epoch: {}'.format(epoch, loss.item()))

        print('loss at last epoch: ', loss.item())

    def accuracy_on_train_set(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.data_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predict = torch.max(outputs, 1)

                correct += (predict == labels).sum().item()
                total += labels.shape[0]

        return correct/total

    @staticmethod
    def draw(w, b):
        x = np.linspace(0, 10, 100)
        y = (w[0] * x + b) / -w[1]

        plt.plot(x, y)

    def visualize(self):
        params = list(self.weights.parameters())

        weight = params[0]
        bias = params[1]

        # print((weight[0] - weight[1]).numpy())
        # print((bias[0] - bias[1]).numpy())

        with torch.no_grad():
            for i in range(0, weight.shape[0] - 1, 1):
                for j in range(i + 1, weight.shape[0], 1):
                    w = weight[i] - weight[j]
                    b = bias[i] - bias[j]
                    w = w.numpy()
                    b = b.numpy()
                    self.draw(w, b)

