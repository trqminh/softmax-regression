from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import matplotlib.pyplot as plt


class DataPoint2DGenerator:
    def __init__(self, mean, cov, num_list):
        self.mean = mean
        self.cov = cov
        self.num_list = num_list  # number data point per class as a list
        self.num_class = len(num_list)
        self.total_points = sum(self.num_list)
        self.data = torch.empty(self.total_points, 2)
        self.labels = torch.empty(self.total_points, dtype=torch.long)

    def generate(self):
        beg = 0
        for i in range(self.num_class):
            torch_mean = torch.tensor(self.mean[i])
            torch_cov = torch.tensor(self.cov)
            distribution = MultivariateNormal(torch_mean, torch_cov)

            if i == 0:
                beg = 0

            end = beg + self.num_list[i]

            for j in range(beg, end, 1):
                self.labels[j] = i
                self.data[j] = distribution.sample()

            beg = end

        result = (self.data, self.labels)
        return result

    def display(self):
        plt.scatter(self.data[:, 0], self.data[:, 1])
