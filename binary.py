import pandas as pd
from sklearn.model_selection import train_test_split

from DecisionTreeBin import *

data_diabetes = pd.read_csv('diabetes.csv', header=None, skiprows=1)
less_data = data_diabetes.sample(60)

y = less_data[8]
X = less_data.drop([8], axis=1)

root = Node()
root.set_is_root()

tree_depth = 7
tree = DecisionTreeBin(root, X, y, tree_depth, X.columns)
tree.build_tree(root)

print_tree(root)

print("Tree overfitted and remember all train selection")
print("Accuracy at tree with depth = 7:" + "\t" + str(count_accuracy(X, y, tree)))

print("Make new tree using all selection:")
y = data_diabetes[8]
X = data_diabetes.drop([8], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
root = Node()
root.set_is_root()

# tree_depth-accuracy
# 1-0.68
# 2-0.73
# 3-0.74
# 4-0.72
# 5-0.71
# 6-0.68
# 7-0.79
tree_depth = 15
tree = DecisionTreeBin(root, X_train, y_train, tree_depth, X.columns)
tree.build_tree(root)

print("Accuracy:" + "\t" + str(count_accuracy(X_test, y_test, tree)))

print("Try this method on other dataset")
data_banknote = pd.read_csv('data_banknote_authentication.txt', header=None)
y = data_banknote[4]
X = data_banknote.drop([4], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
root = Node()
root.set_is_root()

tree_depth = 10
tree = DecisionTreeBin(root, X_train, y_train, tree_depth, X.columns)
tree.build_tree(root)

print("Accuracy:" + "\t" + str(count_accuracy(X_test, y_test, tree)))

