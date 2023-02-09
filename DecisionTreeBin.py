class Node:
    def __init__(self):
        self.feature_value = self.feature = None
        self.right_child = self.left_child = None
        self.class_affiliation = None
        self.indexes_obj_into = []
        self.is_root = False
        self.level = None
        self.dummy = False
        self.unpredictable = False
        self.is_predicate = False

    def set_is_root(self):
        self.is_root = True
        self.level = 1

    def set_dummy(self):
        self.dummy = True

    def set_unpredictable(self):
        self.unpredictable = True

    def transform_into_predicate(self, feature, feature_value):
        self.is_predicate = True
        self.feature = feature
        self.feature_value = feature_value

    def set_left_child(self, node):
        self.left_child = node

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.left_child

    def set_right_child(self, node):
        self.right_child = node

    def set_class_affiliation(self, class_value):
        self.class_affiliation = class_value

    def add_obj_to_node(self, index):
        self.indexes_obj_into.append(index)

    def has_obj(self, index):
        return index in self.indexes_obj_into

    def set_level(self, level):
        self.level = level


class DecisionTreeBin:
    def __init__(self, root_node, train_data, train_target, tree_depth, features):
        self.root = root_node
        self.X = train_data.to_numpy()
        self.tree_depth = tree_depth
        self.y = train_target.to_numpy()
        self.class_number = 2
        self.split_number = 0
        self.features = features

    def build_tree(self, node):
        if node.is_root:
            for i in range(len(self.X)):
                node.add_obj_to_node(i)
        if self.should_stop(node):
            self.select_class(node)
            return
        left, right = self.split(node)
        if node.unpredictable:
            return
        self.build_tree(left)
        self.build_tree(right)

    def should_stop(self, node):
        if node.unpredictable:
            return True
        if node.level >= self.tree_depth:
            return True
        else:
            return False

    def select_class(self, node):
        number_first_class = 0
        number_second_class = 0
        for i in node.indexes_obj_into:
            if self.y[i] == 0:
                number_first_class += 1
            else:
                number_second_class += 1
        if number_first_class > number_second_class:
            node.set_class_affiliation(0)
        else:
            node.set_class_affiliation(1)

    # split using Gini formula 1 - sum(pi)^2. So if Gini = 0 it means that split was good.
    # So we will find split that has best Gini index.
    def split(self, node):
        best_branching_index = None
        best_branching_feature = None
        best_branching_feature_value = None
        indexes = node.indexes_obj_into
        whole_quantity = len(indexes)
        X = self.X
        y = self.y
        for i in self.features:
            for j in indexes:
                right_quantity = 0
                first_quantity_right = 0
                first_quantity_whole = 0
                for k in indexes:
                    if y[k] == 0:
                        first_quantity_whole += 1
                    if X[k][i] > X[j][i]:
                        right_quantity += 1
                        if y[k] == 0:
                            first_quantity_right += 1
                left_quantity = whole_quantity - right_quantity
                first_quantity_left = first_quantity_whole - first_quantity_right
                second_quantity_right = right_quantity - first_quantity_right
                second_quantity_left = left_quantity - first_quantity_left
                second_quantity_whole = whole_quantity - first_quantity_whole
                if right_quantity == 0:
                    information_gain_right = 1
                else:
                    information_gain_right = 1 - pow(first_quantity_right / right_quantity, 2) \
                                             - pow(second_quantity_right / right_quantity, 2)
                if left_quantity == 0:
                    information_gain_left = 1
                else:
                    information_gain_left = 1 - pow(first_quantity_left / left_quantity, 2) \
                                            - pow(second_quantity_left / left_quantity, 2)
                information_gain_whole = 1 - pow(first_quantity_whole / whole_quantity, 2) \
                                         - pow(second_quantity_whole / whole_quantity, 2)
                branching_index = whole_quantity * information_gain_whole - left_quantity * information_gain_left - \
                                  right_quantity * information_gain_right
                if best_branching_index is None:
                    best_branching_index = branching_index
                    best_branching_feature = i
                    best_branching_feature_value = X[j][i]
                elif best_branching_index < branching_index:
                    best_branching_index = branching_index
                    best_branching_feature = i
                    best_branching_feature_value = X[j][i]
        if node.level != self.tree_depth:
            node.transform_into_predicate(best_branching_feature, best_branching_feature_value)
        left = Node()
        right = Node()
        if best_branching_feature_value is None:
            left.set_dummy()
            right.set_dummy()
            node.set_unpredictable()
            return left, right
        left.set_level(node.level + 1)
        right.set_level(node.level + 1)
        right_quantity = 0
        left_quantity = 0
        first_quantity_right = 0
        first_quantity_left = 0
        for i in indexes:
            if X[i][best_branching_feature] > best_branching_feature_value:
                right_quantity += 1
                if y[i] == 0:
                    first_quantity_right += 1
                right.add_obj_to_node(i)
            else:
                left.add_obj_to_node(i)
                left_quantity += 1
                if y[i] == 0:
                    first_quantity_left += 1
        if right_quantity != 0 and first_quantity_right / right_quantity > 0.5:
            right.set_class_affiliation(0)
        else:
            right.set_class_affiliation(1)
        if left_quantity != 0 and first_quantity_left / left_quantity > 0.5:
            left.set_class_affiliation(0)
        else:
            left.set_class_affiliation(1)
        node.set_left_child(left)
        node.set_right_child(right)
        return left, right

    def predict(self, obj, node):
        if node.unpredictable:
            return 0
        if node.is_predicate:
            if obj[node.feature] > node.feature_value:
                return self.predict(obj, node.right_child)
            else:
                return self.predict(obj, node.left_child)
        else:
            return node.class_affiliation


def print_tree(node, level=0):
    if node is not None:
        print_tree(node.left_child, level + 1)
        if node.is_predicate:
            print(' ' * 4 * level + '-> ' + str(len(node.indexes_obj_into)))
        else:
            print(' ' * 4 * level + '-> class : ' + str(node.class_affiliation))
        print_tree(node.right_child, level + 1)


def count_accuracy(X, y, tree):
    correct = 0
    number_all = 0
    for index, row in X.iterrows():
        prediction = tree.predict(row, tree.root)
        number_all += 1
        if prediction == y[index]:
            correct += 1
    return correct / number_all
