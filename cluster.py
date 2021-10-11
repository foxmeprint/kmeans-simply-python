import numpy as np


class clustering:
    def __init__(self, data, core_num, history=False):
        self.data = data
        self.core_num = core_num
        self.history = history
        self.gen = 10
        self.history_data = np.zeros(shape=[self.gen+1,core_num, data.shape[1]])
        self.points_num = data.shape[0]
        indexes = np.random.choice(self.points_num, size=core_num,replace=False)
        self.core = np.zeros(shape=[core_num, data.shape[1]])
        for i, index in enumerate(indexes):
            self.core[i, :] = data[index, :]
        # print('data=\n{}'.format(self.data))
        self.up_his(self.core, 0)
        # print(self.core)
        self.cache = np.zeros(shape=[self.points_num, self.core_num])
        self.shift = np.zeros(self.core_num)
        self.run()

    def run(self):
        for gen in range(self.gen):

            buff = np.zeros(shape=[self.points_num, self.core_num])
            # undate buff
            for i in range(self.core_num):
                for j in range(self.data.shape[1]):
                    buff[:, i] += np.square(self.data[:, j] - self.core[i, j])
            # print('dis\n',buff)
            buff_index = np.argmin(buff, axis=1)
            # print(buff_index)
            for i in range(self.points_num):
                if i ==0:
                    self.shift = np.zeros(self.core_num)
                    self.cache = np.zeros(shape=[self.points_num, self.core_num])
                self.cache[int(self.shift[buff_index[i]]), buff_index[i]] = i
                self.shift[buff_index[i]] += 1
            self.cache = np.array(self.cache, dtype=np.int64)
            # print('cach\n',self.cache)
            # print('shift\n',self.shift)
            self.core = np.zeros(shape=[self.core_num, self.data.shape[1]])
            for i in range(self.core_num):#每一个核心
                for j in range(self.data.shape[1]):#每个核心的每个维度，xyz
                    for k in range(int(self.shift[i])):#每个核心吸引的点个数
                        self.core[i,j] += self.data[self.cache[k, i], j]
                    self.core[i, j]/=self.shift[i]
            if self.history:
                self.up_his(self.core,gen+1)
            # print('core=\n',self.core)
    def up_his(self,core,step):
        self.history_data[step] = np.copy(core)

