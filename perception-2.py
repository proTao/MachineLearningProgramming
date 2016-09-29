# encoding=utf-8

# Perception v0.9：二类感知器算法
#                  没有处理data.txt文件中格式错误的问题


class Perception:
    '''
    感知器算法
    '''

    def __init__(self):
        self.dataset = []  # 存储样本点的增广坐标
        self.w = []  # 存储判决方程的系数向量
        print("欢迎使用感知器分类算法v0.9，只能处理二类分类问题，在data.txt中存放数据，用一个空行分割不同类，单行内的空格分割同一个样本的不同特征值")
        pass

    def getData(self):
        with open("data.txt", "r") as f:
            raw_data = f.readlines()
        flag = False  # 标志着是否读到了空行
        for line in raw_data:
            # 去掉结尾的'\n'
            if line.endswith('\n'):
                line = line[0:-1]
            if len(line) == 0 :
                if len(self.dataset) > 0:
                    flag = True
                else:
                    # 跳过空行
                    continue
            else:
                # 在python3里面
                # map()的返回值已经不再是list,而是iterators
                # 所以想要使用，只用将iterator 转换成list 即可
                # 比如 list(map())
                if flag:
                    line = list(map(lambda x: -int(x), line.split()))
                    line.append(-1)
                else:
                    line = list(map(lambda x: int(x), line.split()))
                    line.append(1)
                self.dataset.append(line)
        length = len(self.dataset[0])
        for i in range(length):
            self.w.append(0)
        print("初始化后的数据为" + str(self.dataset))

    def setW(self):
        # 没有考虑人工输入的向量与读取到的样本点的增广向量的维数不同的情况
        w = input("请输入初始系数向量")
        self.w = list(map(lambda x: int(x), w.split()))
        print(self.w)

    def compute(self):
        is_update = True
        i = 1
        while(is_update):
            print("第%i次迭代" % i)
            is_update = False
            for data in self.dataset:
                if vectorMultiple(data, self.w) <= 0:
                    print(data)
                    print(self.w)
                    print(vectorMultiple(data, self.w))
                    self.w = vectorAdd(self.w, data)
                    print(self.w)
                    print()
                    is_update = True
                else:
                    # 这段对于最终结果的产生是没有用的
                    # 在这里只是为了控制台看看过程
                    print(data)
                    print(self.w)
                    print(vectorMultiple(data, self.w))
                    #self.w = vectorAdd(self.w, data)
                    print(self.w)
                    print()
                    #is_update = False
            i += 1
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


def vectorAdd(v1, v2):
    l1 = len(v1)
    l2 = len(v2)
    result = []
    if l1 == l2:
        for i in range(l1):
            result.append(v1[i] + v2[i])
    else:
        print("输入参数有误")
    return result


p = Perception()
p.getData()
p.compute()
