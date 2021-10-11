import numpy as np
import matplotlib.pyplot as plt
import cluster

fig, ax = plt.subplots()
radius = [5, 5]
points_num = [50, 50]
points = np.array([[1, 1], [5, 6]])

theta1 = np.random.rand(points_num[0]) * np.pi * 2
rho1 = np.random.rand(points_num[0]) * radius[0]

theta2 = np.random.rand(points_num[1]) * np.pi * 2
rho2 = np.random.rand(points_num[1]) * radius[1]

x1 = rho1 * np.cos(theta1) + points[0, 0]
y1 = rho1 * np.sin(theta1) + points[0, 1]
x2 = rho2 * np.cos(theta2) + points[1, 0]
y2 = rho2 * np.sin(theta2) + points[1, 1]

plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'ro')

x = np.concatenate([x1, x2])[:, np.newaxis]
y = np.concatenate([y1, y2])[:, np.newaxis]
data = np.concatenate([x, y], axis=1)
# print(data.shape)
cl = cluster.clustering(data, 2,history=True)
# print(cl.history_data)
plt.plot(cl.history_data[:-1,0,0],cl.history_data[:-1,0,1],'g*-')
plt.plot(cl.history_data[:-1,1,0],cl.history_data[:-1,1,1],'y*-')
plt.plot(cl.history_data[-1,0,0],cl.history_data[-1,0,1],'kx')
plt.plot(cl.history_data[-1,1,0],cl.history_data[-1,1,1],'kx')
# plt.plot(cl.history_data[:,2,0],cl.history_data[:,2,1],'m*-')
plt.show()


