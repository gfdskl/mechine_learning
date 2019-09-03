import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

""" #本来想用这句话去除警告的，但好像没效果，有时间找下原因
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=35)
 """

#加载数据
def load_data(file_dir):
    # iris_data = np.array(pd.read_csv(file_dir,encoding='gbk'))
    iris_data = np.array(pd.read_csv(file_dir))
    # print (iris_data)
    X = iris_data[:,1:5]
    y = iris_data[:,-1]
    #将文本标签转化为数字标签
    y[np.where(y == 'Iris-setosa')] = 1
    y[np.where(y == 'Iris-versicolor')] = 2
    y[np.where(y == 'Iris-virginica')] = 3
    # print (X,y)
    return X,y

#数据预处理，划分训练集和测试集
def preprocess(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
    return X_train,X_test,y_train,y_test

if __name__ == "__main__":
    DIR = 'Iris.csv'
    load_data(DIR)
    [X,y] = load_data(DIR)
    X_train,X_test,y_train,y_test = preprocess(X,y)
    clf = svm.SVC(gamma = 'scale')
    clf.fit(X_train,y_train.astype('int'))
    y_predict = clf.predict(X_test)
    k = (np.argwhere(y_test==y_predict))
    accuracy_scor = k.shape[0]/len(y_test)*100
    print ('准确度为：%.2f%%' % accuracy_scor)