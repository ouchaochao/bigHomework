from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydot


if __name__ == '__main__':
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    dt = tree.DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    print("决策树：",dt.score(x_test, y_test))

    dot_data = StringIO()
    tree.export_graphviz(dt, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_dot('iris_simple.dot')
    graph[0].write_png('iris_simple.png')