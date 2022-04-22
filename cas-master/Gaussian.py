import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

def Gaussian():
        # Build a model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Some data
    xobs = np.array([[1], [1.5], [-3]])
    yobs = np.array([3, 0, 1])

    # Fit the model to the data (optimize hyper parameters)
    gp.fit(xobs, yobs)

    # Plot points and predictions
    x_set = np.arange(-6, 6, 0.1)
    x_set = np.array([[i] for i in x_set])
    means, sigmas = gp.predict(x_set, return_std=True)
    # 预测完成

'''
plt.figure(figsize=(8, 5))
plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5)
plt.plot(x_set, means, 'g', linewidth=4)

colors = ['g', 'r', 'b', 'k']
for c in colors:
    y_set = gp.sample_y(x_set, random_state=np.random.randint(1000))
    plt.plot(x_set, y_set, c + '--', alpha=0.5)

plt.show()
'''