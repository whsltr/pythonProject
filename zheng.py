import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def f(x, eta):
    return x**2 / (1 + x*x -eta)**(k+1/2)


for k in [2, 3, 4, 5, 7, 10, 50]:
    A = []
    space = np.linspace(0, 3, 100)
    for phi in space:
        eta = 2 / (2 * k - 3) * phi
        ymin = np.sqrt(eta)
        ymax = 100
        P1 = ((ymax - ymin) / 3000) * (f(ymin, eta) + f(ymax, eta) + 4 * sum(
            f(ymin + (2 * j - 1) * ((ymax - ymin) / 1000), eta) for j in range(1, 501))
                                      + 2 * sum(
                    f(ymin + (2 * j) * ((ymax - ymin) / 1000), eta) for j in range(1, 500)))
        a = 4 * P1 * gamma(k + 0.5) / (np.sqrt(np.pi) * gamma(k - 1))
        A.append(a)

    A = np.array(A)
    phi = np.linspace(0, 4, 50)

    plt.plot(space, A, label=r'$\frac{e\phi}{k_b T_e}=$' + str(k))
plt.legend()
plt.show()