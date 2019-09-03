from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import random

#加载数据
def load_data(file_dir):
    m = loadmat(file_dir)
    X = np.array(m['X'])
    return X

#画图
def paint_X(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()

def kmeans(X,k):
    #初始化中心点
    time = 0
    flag = 1
    sub = np.arange(len(X))
    random.shuffle(sub)
    sub = sub[:k]
    center = X[sub]
    c_last = np.empty((len(X),),dtype = 'int')
    #计算每个点离最近中心点的下标,用c数组存储
    while (flag == 1):
        for i in range(len(X)):
            p = np.sum((X[i]-center)*(X[i]-center),axis = 1)
            nearest_sub = np.argwhere(p == min(p))
            c_last[i] = nearest_sub[0][0]

        #计算新的中心点
        sub_ = 0
        c_unipue = np.unique(c_last).tolist()
        k = len(c_unipue)
        new_center = np.empty((k,2),dtype = 'float')
        for i in c_unipue:
            new_center[sub_] = np.mean(X[np.argwhere(c_last == i)])
            sub_ += 1
                
        if (new_center.shape == center.shape and (new_center == center).all() == True):
            flag = 0
        center = new_center
        cost = cost_function(X,c_last,center)
        print ('time:%d cost:%.2f'%(time,cost))
        time += 1
    return c_last,k,center

    #画出最后的分割图
def plot_ans(c_last,k):
    sub = [0 for i in range(k)]
    for i in range(k):
        sub[i] = np.argwhere(c_last == i)
        sub[i] = sub[i].reshape(len(sub[i],))
    plt.scatter(X[sub[0]][:,0],X[sub[0]][:,1],c = 'g')
    plt.scatter(X[sub[1]][:,0],X[sub[1]][:,1],c = 'r')
    plt.scatter(X[sub[2]][:,0],X[sub[2]][:,1])
    # plt.scatter(X[sub[3]][:,0],X[sub[3]][:,1],c = 'y')
    plt.show()

def cost_function (X,c_last,center):
    J = ((X-center[c_last])*(X-center[c_last])).sum()
    return J


if __name__ == "__main__":
    # DIR = 'ex7data1.mat'
    DIR = 'ex7data2.mat'
    X = np.array(load_data(DIR))
    # paint_X(X)
    k = 3
    # k = int(input('the number of classes:'))
    c_last,k,center = kmeans(X,k)
    plot_ans(c_last,k)
