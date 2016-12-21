# encoding=utf-8

# Perception v0.9：二类感知器算法
# 每个神经节点的各个输入线性组合
# 然后输入激活函数进行输出

# 当激活函数TF使用sigmoid，误差函数使用sig_error时，就是多层前馈网络中后向传播的节点
import numpy as np
from math import fabs, exp
from random import randint


class Perception:
    '''
    感知器算法
    '''

    def __init__(self, max_iter, TF, eps, min_error, EF):
        self.dataset = []  # 存储样本点的增广坐标,这里用列表初始化，但是在getData过后会转变成matrix类型
        self.label = []  # 以list形式存储
        self.w = []  # 存储判决方程的系数向量,w中的第一个元素是b，对应着，所有数据向量第一个位置都是1
        self.max_iter = max_iter  # 最大迭代次数
        self.TF = TF  # 节点的激活函数
        self.eps = eps  # 学习速率
        self.min_error = min_error  # 误差阈值，需要匹配激活函数
        self.EF = EF  # 误差函数，实际上是通过梯度求出来的
        # print("欢迎使用感知器分类算法v0.9，只能处理二类分类问题，在data.txt中存放数据，用一个空行分割不同类，单行内的空格分割同一个样本的不同特征值")
        pass

    def getData(self, file):
        #
        with open(file, "r") as f:
            raw_data = f.readlines()
        for line in raw_data:
            # 去掉结尾的'\n'
            if line.endswith('\n'):
                line = line[0:-1]

            # 在python3里面
            # map()的返回值已经不再是list,而是iterators
            # 所以想要使用，只用将iterator 转换成list 即可
            # 比如 list(map())
            if(line):
                line = list(map(lambda x: float(x), line.split()))
                data = [1]  # 偏置量
                data.extend(line)
                self.dataset.append(data[:-1])
                self.label.append(data[-1])

        # 自动将连接权重赋值为零
        length = len(self.dataset[0])
        self.w = np.mat([0.0] * length).T

        self.dataset = np.mat(self.dataset)
        self.w = np.mat(self.w).T
        print(self.dataset)
        print(self.label)
        print(self.w)
        print()

    def changeW(self, w):
        self.w = np.mat(w).T

    def getW(self):
        return self.w

    def output(self, data):
        # 乘出来的结果是一个1行1列的矩阵，所以用[0,0]取其中元素
        y_hat = self.TF((data * self.w)[0, 0])
        return y_hat

    def computeError_o(self, y_hat, label):
        # 输出层的步骤，计算误差并回传
        error = EF(label, y_hat)
        return error

    def computeError_h(self, y_hat, weights, errors):
        # 隐藏层的计算误差
        # weights是与下一层节点连接的权重，errors是对应的误差
        error = 0
        for i in range(len(weights)):
            error += weights[i] * errors[i]
        return y_hat * (1 - y_hat) * error

    # 需要改动
    def updateW(self, label):
        # 给出一组数据和该组数据的label，进行更新
        # 乘出来的结果是一个1行1列的矩阵，所以用[0,0]取其中元素
        y_hat = self.TF((data * self.w)[0, 0])
        error = self.EF(label, y_hat)
        '''
        print(self.w.T)
        print(data)
        print("yhat:"+str(y_hat))
        print("label:"+str(label))
        print("error:"+str(error))
        '''
        if(fabs(error) > self.min_error):
            self.w += self.eps * error * data.T
            print(self.w.T)
            print()
            return True
        else:
            print()
            return False

    def compute(self):
        for i in range(self.max_iter):
            is_update = False

            for j in range(len(self.dataset)):
                is_update = self.update(
                    self.dataset[j], self.label[j]) or is_update
            '''
            j=randint(0,3)
            is_update = self.update(self.dataset[j],self.label[j]) or is_update
            '''
            error = 0
            '''
            temp=0
            for i in self.w:
                temp += i[0,0] **2
            self.w = self.w/(temp**0.5)
            '''
            for j in range(len(self.dataset)):
                error += (self.dataset[j] * self.w - self.label[j]) ** 2
            print("***************%f**************" % error)
            if(is_update == False):
                break


def hardlim(x):
    if(x > 0):
        return 1
    elif(x == 0):
        return 0
    else:
        return -1


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + exp(-x))


def error1(label, y):
    return label - y


def sig_error(label, y):
    return y * (1 - y) * (label - y)

'''
p = Perception(1,sigmoid,0.9,0,sig_error)
p.getData("data.txt")
p.changeW([0.1,-0.3,-0.2])
p.compute()
'''


class BPnetwork:

    def __init__(self, inputnum, hiddennum, outputnum):
        self.inputnum = inputnum
        self.hiddennum = hiddennum
        self.outputnum = outputnum
        self.hidden_layer=[]
        self.output_layer=[]
        self.data=[1,0,1]
        for i in range(hiddennum):
            self.hidden_layer.append(Perception(100,sigmoid,0.9,0,sig_error))
        for i in range(outputnum):
            self.output_layer.append(Perception(1,sigmoid,0.9,0,sig_error))

    def compute(self):
        for p in hidden_layer:
            p.output(self.data)