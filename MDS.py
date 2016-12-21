import numpy as np

distance = [[0, 411, 213, 219, 296, 397],
            [411, 0, 204, 203, 120, 152],
            [213, 204, 0, 73, 136, 245],
            [219, 203, 73, 0, 90, 191],
            [296, 120, 136, 90, 0, 109],
            [397, 152, 245, 191, 109, 0]]

for i in range(6):
    for j in range(6):
        distance[i][j] = distance[i][j]**2
def makeB(mat):
    # B矩阵是人工构造的矩阵
    length = len(mat[0])
    di = []
    for i in range(length):
        di.append(sum(mat[i]))
    d = sum(di)
    B = []
    for i in range(length):
        temp = []
        for j in range(length):
            temp.append(makeBij(mat[i][j], di[i], di[j], d, length))
        B.append(temp)
    return B


def makeBij(dij, di, dj, d, n):
    # n是方阵的边长
    # 输入参数。并对B矩阵中的元素进行计算
    return -0.5 * (dij - di / n - dj / n + d / (n**2))


def getDistance(p1, p2, dim):
    # p1和p2都是numpy的矩阵元素，所以要转化成list在进行计算
    distance=0
    for i in range(dim):
        distance+=(p1.tolist()[0][i] - p2.tolist()[0][i])**2
        # distance += p1.tolist()[0][i] * p2.tolist()[0][i]
    return distance**0.5
    


def makeX(B, dim):
    # 还原X矩阵，X的每一行都是一个还原出来的点
    # dim是想还原到的维度，在MDS算法中只能取值为2或者3
    # 对B矩阵进行特征值分解，x是特征值数组，y是特征向量构成的矩阵，每一列是一个特征值
    x, y = np.linalg.eig(B)
    

    features = []
    x = x.tolist()
    y = np.mat(y).T.tolist()

    # feature中的每一个元素都是一个特征值和它对应的特征向量
    for i in range(len(x)):
        features.append([x[i], y[i]])

    # 关于特征值排序
    features = sorted(features, key=lambda feature: feature[0], reverse=True)

    # 取排序后前dim个的特征值及其对应的矩阵
    features = features[0:dim]

    x = []
    y = []
    for feature in features:
        x.append(feature[0]**0.5)
        y.append(feature[1])
   
    X=np.mat(np.diag(x)) * np.mat(y)
    return X.T

def validateDistance(X):
    # X的每一行都是一个点，
    # X是numpy的矩阵类型
    length=len(X)
    distance=[]
    dim = len(X[0].tolist()[0])
    for i in range(length):
        temp=[]
        for j in range(length):
            temp.append(getDistance(X[i],X[j],dim))
        distance.append(temp)
    return distance

B = makeB(distance)
print(np.mat(B))
X = makeX(B, 3)

print(X*X.T)
D=validateDistance(X)
print(np.mat(D))
