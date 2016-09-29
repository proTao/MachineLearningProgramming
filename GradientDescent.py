# encoding=utf-8

# Perception v1.0：多类感知器算法
#                  没有处理data.txt文件中格式错误的问题


class Perception:
    '''
    感知器算法
    '''

    def __init__(self):
        self.dataset = []   # 存储样本点的增广坐标
        # 和perceptionv0.9不同，这里的dataset是3维数组
        # 详情见getData函数
        self.w = []  # 存储判决方程的系数向量
        # 和perceptionv0.9不同，这里的w是2维数组
        self.sample_num = 0  # 共有多少个训练样本
        self.class_num = 0  # 数据分为多少类
        pass

    def getData(self):
        with open("data.txt", "r") as f:
            raw_data = f.readlines()

        temp_class = []
        # 暂时存储读取到的一类中的样本
        # 读取到下一个类的样本时，
        # 将该类中的值一次性追加入dataset

        for line in raw_data:
            # 去掉结尾的'\n'
            if line.endswith('\n'):
                line = line[0:-1]
            if len(line) == 0:
                if len(temp_class) != 0:
                    self.class_num += 1
                    self.dataset.append(temp_class)
                    temp_class = []
                else:
                    # 跳过空行
                    continue
            else:
                # 在python3里面
                # map()的返回值已经不再是list,而是iterators
                # 所以想要使用，只用将iterator 转换成list 即可
                # 比如 list(map())
                line = list(map(lambda x: int(x), line.split()))
                line.append(1)
                temp_class.append(line)
                self.sample_num += 1
        if len(temp_class) != 0:
            self.class_num += 1
            self.dataset.append(temp_class)
            temp_class = []

        length = len(self.dataset[0][0])    # 每一个样本中的特征数 即维度
        temp_w = []
        # 把self.w初始化为一个有class_num行，length列的矩阵
        # 第i行即为第i类的系数向量
        for j in range(self.class_num):
            for i in range(length):
                temp_w.append(0)
            self.w.append(temp_w)
            temp_w = []
        print(self.dataset)
        print(self.w)

    def compute(self):
        d = []
        for i in range(self.class_num):
            d.append(0)
        no_update_count = 0  # 表示连续几次迭代没有更新数据
        count = 1   # 已经运行了几次
        run = True  # 是否需要继续运行，由no_update_count控制
        while run:
            for i in range(len(self.dataset)):
                # 在处理第i类类别中
                for data in self.dataset[i]:
                    # data是第i类别中的数据
                    if no_update_count < self.sample_num:
                        # 系数还没有稳定，需要继续运行
                        print("第%i次迭代" % count)
                        count += 1
                        flag = False
                        for j in range(self.class_num):
                            # 第一次循环，计算当前系数下，所有判别函数的值
                            d[j] = vectorMultiple(self.w[j], data)

                        print("d" + str(d))
                        for j in range(self.class_num):
                            # 第二次循环，根据判别函数的值，进行系数调整
                            if i == j:
                                continue
                            else:
                                if d[i] <= d[j]:
                                    self.w[j] = vectorLinearCombination(
                                        self.w[j], data, 1, -1)
                                    flag = True
                        if flag:
                            # 说明该次循环进行了更新
                            self.w[i] = vectorLinearCombination(
                                self.w[i], data, 1, 1)
                            no_update_count = 0
                        else:
                            no_update_count += 1
                        print("no_update_count" + str(no_update_count))
                        print(self.w)
                    else:
                        # 遍历过一次样本且系数没有变化，系数稳定，停止运行
                        run = False
        print("最终结果为" + str(self.w))


def vectorMultiple(v1, v2):
    l1 = len(v1)
    l2 = len(v2)
    result = 0
    if l1 == l2:
        for i in range(l1):
            result += v1[i] * v2[i]
    else:
        print("输入参数有误")
    return result


def vectorLinearCombination(v1, v2, a=1, b=1):
    # 以a*v1+b*v2的方式将两个向量进行线性组合
    l1 = len(v1)
    l2 = len(v2)
    result = []
    if l1 == l2:
        for i in range(l1):
            result.append(a * v1[i] + b * v2[i])
    else:
        print("输入参数有误")
    return result


p = Perception()
p.getData()
p.compute()
