from datapoint_generator import DataPoint2DGenerator
from softmax_regression import SoftmaxRegression
from my_data import MyDataset
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse

if __name__ == "__main__":
    np.random.seed(0), torch.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=0, help='1 for training and 0 for testing')
    args = parser.parse_args()

    if bool(args.train):
        print("Trainning...")
        means = [[2., 2.], [-2., -2.], [-5., 6.]]
        cov = [[1., 0.], [0., 1.]]

        data_generator = DataPoint2DGenerator(means, cov)
        data = data_generator.generate()
        data_generator.display()

        dataset = MyDataset(data[0], data[1])
        soft_reg = SoftmaxRegression(dataset, data_generator.n_class)
        soft_reg.train()

        # accuracy on train set
        soft_reg.visualize()
        print('Accuracy on train set: ', soft_reg.accuracy_on_train_set())

        # show all plot
        plt.show()
    else:
        print("Visualization: ")
        means = [[2., 2.], [-2., -2.], [-5., 6.]]
        cov = [[1., 0.], [0., 1.]]

        data_generator = DataPoint2DGenerator(means, cov)
        data = data_generator.generate()
        data_generator.display()

        dataset = MyDataset(data[0], data[1])
        soft_reg = SoftmaxRegression(dataset, data_generator.n_class)

        # accuracy on train set
        soft_reg.visualize(True)

        # show all plot
        plt.show()
