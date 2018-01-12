from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn import tree
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def read_csv():
    # read_csv
    global X, Y, X1, Y1
    df_train = pd.read_csv('E:\ML\Training-set.csv')
    df_test = pd.read_csv('E:\ML\Testing-set-label.csv')
    train = df_train.values
    test = df_test.values
    X = train[:, 1:4]
    X1 = train[:, 4]
    Y = test[:, 1:4]
    Y1 = test[:, 4]


def draw2(c, mothod):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], label=mothod, c=c)
    ax.legend()
    plt.show()


def d_tree():
    read_csv()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, X1)
    score = round(clf.score(Y, Y1), 3)
    print('决策树正确率(accuracy):', score)
    # draw tree
    c = clf.predict(Y)
    draw2(c, 'tree')


def k_nn():
    read_csv()
    knn = neighbors.KNeighborsClassifier(n_neighbors=19, weights='distance')
    knn = knn.fit(X, X1)
    score = round(knn.score(Y, Y1), 3)
    print('knn正确率(accuracy):', score)
    # draw knn
    c = knn.predict(Y)
    draw2(c, 'knn')


# 支持向量机
def svm():
    read_csv()
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, X1)
    score = round(clf.score(Y, Y1), 3)
    print('支持向量机正确率(accuracy):', score)
    c = clf.predict(Y)
    draw2(c, 'svm')


# 朴素贝叶斯
def n_b():
    read_csv()
    gnb = GaussianNB()
    gnb.fit(X, X1)
    score = round(gnb.score(Y, Y1), 3)
    print('朴素贝叶斯正确率(accuracy):', score)
    c = gnb.predict(Y)
    draw2(c, 'NB')


if __name__ == '__main__':
    svm()
    n_b()
    d_tree()
    k_nn()
