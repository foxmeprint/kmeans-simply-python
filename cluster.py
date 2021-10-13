import numpy as np
import matplotlib.pyplot as plt


# 我会尽力讲明白这个py文件究竟干了什么

class clustering:
    '''
    __init__收集数据集，核心数以及是否需要记录历史（默认是false）
    self.gen表示迭代的最大次数，默认10，所以计算核心次数为10，这是为了适应二维的数据比较简单
    self.history_data记录所有历史核心的参数，在二维平面，表示为坐标
    self.data表示为原数据，为一个numpy矩阵，行代表数据点的个数，列代表每个数据点的维度
    '''

    def __init__(self, data, core_num, history=False):
        self.data = data
        self.core_num = core_num
        self.history = history
        self.gen = 10
        self.history_data = np.zeros(shape=[self.gen + 1, core_num, data.shape[1]])
        self.points_num = data.shape[0]
        # 在所有数据点中选择corenum个不同的数据点作为初始的核心
        indexes = np.random.choice(self.points_num, size=core_num, replace=False)
        self.core = np.zeros(shape=[core_num, data.shape[1]])
        for i, index in enumerate(indexes):
            self.core[i, :] = data[index, :]
        # 更新历史记录
        self.up_his(self.core, 0)
        #
        self.cache = np.zeros(shape=[self.points_num, self.core_num])
        self.shift = np.zeros(self.core_num)
        self.run()

    def run(self):
        for gen in range(self.gen):
            # buff矩阵储存每个数据点到每一个核心的代价，在二维世界这里是距离（平方和）
            # 它的每一行代表一个数据点
            buff = np.zeros(shape=[self.points_num, self.core_num])
            # undate buff
            for i in range(self.core_num):
                for j in range(self.data.shape[1]):
                    buff[:, i] += np.square(self.data[:, j] - self.core[i, j])
            # 找到每一行中值最小的对应的列，意思是这行的数据点选择了这一列对应的核心
            # 得到一个序列buff_index，长度为数据点数，值为可能的核心名称
            buff_index = np.argmin(buff, axis=1)
            '''
            统计每个核心所领导的数据点
            遍历buff_index，将该数据点编号记录在对应核心名单下，得到cache矩阵
            这看起来更像一个直方图
            因为这个矩阵很大，所以总会有零，这些零对以后的计算不会有影响
            shfit序列帮助记录每个核心领导的数据点个数
            '''
            for i in range(self.points_num):
                if i == 0:
                    self.shift = np.zeros(self.core_num)
                    self.cache = np.zeros(shape=[self.points_num, self.core_num])
                self.cache[int(self.shift[buff_index[i]]), buff_index[i]] = i
                self.shift[buff_index[i]] += 1
            self.cache = np.array(self.cache, dtype=np.int64)
            '''
            根据得到的核心领导的名单更新核心的新位置
            每个核心领导的数据点的每一维参数相加，平均
            根据kmean算法的意义，这个算数平均就是新的核心的位置了
            选择性地记录历史
            '''
            self.core = np.zeros(shape=[self.core_num, self.data.shape[1]])
            for i in range(self.core_num):  # 每一个核心
                for j in range(self.data.shape[1]):  # 每个核心的每个维度，xyz
                    for k in range(int(self.shift[i])):  # 每个核心吸引的点个数
                        self.core[i, j] += self.data[self.cache[k, i], j]
                    self.core[i, j] /= self.shift[i]
            if self.history:
                self.up_his(self.core, gen + 1)

    def up_his(self, core, step):
        self.history_data[step] = np.copy(core)


if __name__ == "__main__":
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
    cl = clustering(data, 2, history=True)
    # print(cl.history_data)
    plt.plot(cl.history_data[:-1, 0, 0], cl.history_data[:-1, 0, 1], 'g*-')
    plt.plot(cl.history_data[:-1, 1, 0], cl.history_data[:-1, 1, 1], 'y*-')
    plt.plot(cl.history_data[-1, 0, 0], cl.history_data[-1, 0, 1], 'kx')
    plt.plot(cl.history_data[-1, 1, 0], cl.history_data[-1, 1, 1], 'kx')
    # plt.plot(cl.history_data[:,2,0],cl.history_data[:,2,1],'m*-')
    plt.show()
