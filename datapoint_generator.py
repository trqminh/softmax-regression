from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import matplotlib.pyplot as plt


class DataPoint2DGenerator:
    def __init__(self, means, cov, n_point_per_class=100):
        self.means = means
        self.cov = cov
        self.n_point_per_class = n_point_per_class
        self.n_class = len(means)
        self.total_points = self.n_class * self.n_point_per_class
        self.data = torch.empty(self.total_points, 2)
        self.labels = torch.empty(self.total_points, dtype=torch.long)

    def generate(self):
        point_idx = 0
        for i in range(self.n_class):
            torch_mean = torch.tensor(self.means[i])
            torch_cov = torch.tensor(self.cov)
            distribution = MultivariateNormal(torch_mean, torch_cov)

            for _ in range(self.n_point_per_class):
                self.labels[point_idx] = i
                self.data[point_idx] = distribution.sample()
                point_idx += 1

        result = (self.data, self.labels)
        return result

    def display(self):
        plt.scatter(self.data[:, 0], self.data[:, 1])
