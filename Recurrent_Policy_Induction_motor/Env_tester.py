import numpy as np
import matplotlib.pyplot as plt

one = np.random.randint(-5, 5, 1000)
two = one * one
four = np.power(one, 5)
print(two.shape)

plt.plot(one)
# plt.plot(one)
# # plt.plot(one, one)

plt.show()