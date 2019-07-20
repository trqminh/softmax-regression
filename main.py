from datapoint_generator import DataPoint2DGenerator
from softmax_regression import SoftmaxRegression
from my_data import MyDataset
import matplotlib.pyplot as plt

mean = [[2., 2.], [8., 3.], [3., 6.]]
cov = [[1., 0.], [0., 1.]]

data_generator = DataPoint2DGenerator(mean, cov, [10, 10, 10])
data = data_generator.generate()
data_generator.display()

dataset = MyDataset(data[0], data[1])

soft_reg = SoftmaxRegression(dataset, data_generator.num_class)
soft_reg.train()

# accuracy on train set
print('accuracy on train set: ', soft_reg.accuracy_on_train_set())
soft_reg.visualize()

# show all plot
plt.show()
