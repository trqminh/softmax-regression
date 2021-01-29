from datapoint_generator import DataPoint2DGenerator
from softmax_regression import SoftmaxRegression
from my_data import MyDataset
import matplotlib.pyplot as plt
import torch
import numpy as np

np.random.seed(0), torch.manual_seed(0)

means = [[2., 2.], [-2., -2.], [-5., 6.]]
cov = [[1., 0.], [0., 1.]]

data_generator = DataPoint2DGenerator(means, cov)
data = data_generator.generate()
data_generator.display()

dataset = MyDataset(data[0], data[1])

soft_reg = SoftmaxRegression(dataset, data_generator.n_class)
# soft_reg.train()

# accuracy on train set
# print('Accuracy on train set: ', soft_reg.accuracy_on_train_set())
soft_reg.visualize(True)

# show all plot
plt.show()
