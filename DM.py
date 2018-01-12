import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 对实验数据进行处理
def pretreatment():
    # read_csv
    df_train = pd.read_csv('E:\kaggle\Datasets/Breast-Cancer/breast-cancer-train.csv')
    df_test = pd.read_csv('E:\kaggle\Datasets/Breast-Cancer/breast-cancer-test.csv')
    # 正负分类样本，特征：肿块厚度和细胞大小
    df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
    df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]
    return df_test_negative, df_test_positive, df_train, df_test


def mse(targe, predictions):
    squared_deviation = np.power(targe - predictions, 2)
    return np.mean(squared_deviation)


# 线性回归分类
def liner():
    df_test_negative, df_test_positive, df_train, df_test = pretreatment()

    lx = np.arange(0, 12)
    lr = LogisticRegression()
    # 使用所有样本学习直线的系数和截距
    lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
    score = round(lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']), 3)
    print('线性正确率 (accuracy):', score)
    # print('MSE: ', mse(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

    intercept = lr.intercept_
    coef = lr.coef_[0, :]
    ly = (-intercept - lx * coef[0]) / coef[1]
    # 绘制图1-5
    plt.plot(lx, ly, c='blue')
    plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()


# 支持向量机
def svm():
    df_test_negative, df_test_positive, df_train, df_test = pretreatment()
    clf = SVC(kernel='linear', C=1.0)
    X = df_train[['Clump Thickness', 'Cell Size']]
    y = df_train['Type']
    clf.fit(X, y)
    score = round(clf.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']), 3)
    print('支持向量机正确率(accuracy):', score)

    w = clf.coef_[0]
    a = -w[0] / w[1]  # a可以理解为斜率
    xx = np.linspace(-5, 5)
    yy = a * xx - clf.intercept_[0] / w[1]  # 二维坐标下的直线方程
    plt.plot(xx, yy, c='red')
    plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()


# 朴素贝叶斯
def n_b():
    df_test_negative, df_test_positive, df_train, df_test = pretreatment()
    X = df_train[['Clump Thickness', 'Cell Size']]
    y = df_train['Type']
    gnb = GaussianNB()
    gnb.fit(X, y)
    score = round(gnb.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']), 3)
    print('朴素贝叶斯正确率(accuracy):', score)


if __name__ == '__main__':
    liner()
    svm()
    n_b()
