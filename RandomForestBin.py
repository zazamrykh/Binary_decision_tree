import random
from DecisionTreeBin import *


class RandomForestBin:
    def __init__(self, train_data, train_target, trees_number, feature_number, tree_depth):
        self.trees_number = trees_number
        self.feature_number = feature_number
        self.tree_depth = tree_depth
        self.forest = []
        for i in range(trees_number):
            root = Node()
            root.set_is_root()
            tree = DecisionTreeBin(root, train_data, train_target, tree_depth,
                                   random.choices(train_data.columns, k=feature_number))
            tree.build_tree(root)
            self.forest.append(tree)

    def predict(self, obj):
        first_class_predictions = 0
        second_class_predictions = 0
        for tree in self.forest:
            prediction = tree.predict(obj)
            if prediction == 0:
                first_class_predictions += 1
            else:
                second_class_predictions += 1
        if first_class_predictions >= second_class_predictions:
            return 0
        else:
            return 1
