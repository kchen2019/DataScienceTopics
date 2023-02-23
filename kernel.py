import pandas as pd
# df
label = [1,1,1,1,1,0,0,0,0,0,1,1,1,1]
samples = [1,2,3,4,5,6,7,8,9,10,11,12, 13, 14]

df = pd.DataFrame([samples, label]).transpose()
df.columns = ['samples', 'label']

import matplotlib.pyplot as plt
import numpy as np

# plot them
plt.rcParams["figure.figsize"] = [5, 5]
y_value = 0
y = np.zeros_like(samples) + y_value
plt.scatter(samples, y, ls='dotted', c=df['label'])
# plt.scatter(samples, y, ls='dotted')
plt.show()

# polynomial kernel
def poly_k(x):
    return (x-8)**2
def poly_k_new(x):
    return (x.transpose()*x-8)**2

#1d
# exp_k_sample = [poly_k(x) for x in samples]
exp_k_sample = (np.array(samples).transpose()*samples-8)**2
# plot them
plt.scatter(samples, exp_k_sample, ls='dotted', c=df['label'])
plt.show()

# 2d
# Creating equally spaced 100 data in range 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 50)
# Setting radius
radius1 = 5
# Generating x and y data
x1 = radius1 * np.cos(theta)
y1 = radius1 * np.sin(theta)
radius2 = 3
x2 = radius2 * np.cos(theta)
y2 = radius2 * np.sin(theta)
# Plotting
plt.plot(x1, y1, ls='dotted', color = 'red')
plt.plot(x2, y2, ls='dotted', color = 'blue')
plt.show()

def circle_kernel(x, y):
    return x**2+y**2

z2 = list(map(lambda x, y: circle_kernel(x, y), x2, y2))
z1 = list(map(lambda x, y: circle_kernel(x, y), x1, y1))

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(x1, y1, z1, c = 'r', s = 50)
ax.scatter(x2, y2, z2, c = 'b', s = 50)
plt.show()


import math
def bivariate_kernel(x, y, xmean, ymean, xstd, ystd):
    pi = math.pi
    ratio = 1/(2*pi*xstd*ystd)
    standard_x = ((x-xmean)/xstd)**2
    standard_y = ((y-ymean)/ystd)**2
    f_xy = np.exp(-1/2*(standard_x+standard_y))
    return ratio*f_xy

z2 = list(map(lambda x, y: circle_kernel(x, y), x2, y2))
z1 = list(map(lambda x, y: circle_kernel(x, y), x1, y1))

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(x1, y1, z1, c = 'r', s = 50)
ax.scatter(x2, y2, z2, c = 'b', s = 50)
plt.show()

x_all = list(x1) + list(x2)
y_all = list(y1) + list(y2)

x_mean = np.mean(x_all)
y_mean = np.mean(y_all)
x_std = np.std(x_all)
y_std = np.std(y_all)

z_all = list(map(lambda x, y: bivariate_kernel(x, y, x_mean, y_mean, x_std, y_std), x_all, y_all))

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter(x_all, y_all, z_all, c = 'r', s = 50)
plt.show()


from scipy.stats import multivariate_normal
#Create grid and multivariate normal
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
rv = multivariate_normal([x_mean, y_mean], [[x_std, 0], [0, y_std]])

pos1 = np.empty((50,50) + (2,))
pos1[:, :, 0] = x1
pos1[:, :, 1] = y1
big_circle =rv.pdf(pos1)

pos2 = np.empty((50,50) + (2,))
pos2[:, :, 0] = x2
pos2[:, :, 1] = y2
small_circle =rv.pdf(pos2)


#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='RdYlGn',linewidth=0, alpha=0.5)
# ax.scatter(x1, y1, big_circle, ls='dotted', color = 'red')
ax.scatter(x_all, y_all, z_all, c = 'blue', s = 50)
# ax.scatter(x2, y2, small_circle, ls='dotted', color = 'blue')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()