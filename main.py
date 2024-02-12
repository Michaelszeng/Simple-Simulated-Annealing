import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
    
fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection": "3d"})

# Generate Random Optimization Landscape
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
rand1 = np.random.rand(1)*0.5 + 0.5
rand1_1 = np.random.rand(1)*0.5 + 0.5
rand1_2 = np.random.rand(1)*0.5 + 0.5
rand2 = np.random.rand(1)*0.5 + 0.5
rand2_1 = np.random.rand(1)*0.5 + 0.5
rand2_2 = np.random.rand(1)*0.5 + 0.5
z_func = lambda x, y: 0.5*rand1*np.sin(rand1_1*x)*np.cos(rand1_2*x) + rand2*np.sin(rand2_1*y)*np.cos(rand2_2*y)
Z = z_func(X, Y)
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, alpha=0.5, antialiased=False)
plt.ion()


current_xy = np.random.uniform(-3.0, 3.0, (2,))  # random X,Y start
current_z = z_func(*current_xy)

temp = 100

while (temp >= 1):
    while True:
        random_perturb = np.random.uniform(0.1, 0.25, (2,)) * np.array([np.random.choice([-1, 1]), np.random.choice([-1, 1])]) 
        print(current_xy+random_perturb)
        if np.all((current_xy+random_perturb) <= 3.0)  and np.all((current_xy+random_perturb) >= -3.0):
            break

    perturbed_z = z_func(*(current_xy+random_perturb))

    delta_z = perturbed_z - current_z
    ran = np.random.rand(1)
    # Probability of taking a step up the gradient
    # Note: if perturbed_z < current_z, the below if statement will always evaluate to True.
    if ran <= math.e**(-delta_z/(temp/100)):
        current_xy = current_xy+random_perturb
        current_z = perturbed_z

    print(f"({temp} deg: {current_xy[0]}, {current_xy[1]}, {current_z[0]})")

    ax.scatter(*current_xy, current_z, color='red', s=10)  # s is the size of the dot
    plt.pause(0.01)

    # Lower temp
    temp -= 0.5

print("DONE.")
ax.scatter(*current_xy, current_z, color='blue', s=25)  # s is the size of the dot

plt.show(block=True)