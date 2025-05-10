import numpy as np
import matplotlib.pyplot as plt

### ID1: 207640822
### ID2: 205965437

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]
    p_arr = np.unique(labels, return_counts=True)[1] / len(data)
    gini = 1 - (p_arr ** 2).sum()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:, -1]
    p_arr = np.unique(labels, return_counts=True)[1] / len(data)
    entropy = - (p_arr * np.log2(p_arr)).sum()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False, n_total_sample=None):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
        self.n_total_sample = n_total_sample if n_total_sample is not None else len(data)

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = self.data[:, -1]
        values, counts = np.unique(labels, return_counts=True)
        pred = values[np.argmax(counts)]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.terminal or not self.children:
            self.feature_importance = 0
            return

        weighted_parent_impurity = (len(self.data) / n_total_sample) * self.impurity_func(self.data)

        weighted_children_impurity = sum(
            (len(child.data) / n_total_sample) * self.impurity_func(child.data)
            for child in self.children
        )

        self.feature_importance = weighted_parent_impurity - weighted_children_impurity
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        feature_values = self.data[:, feature]
        unique_values = np.unique(feature_values)
        groups = {val: self.data[feature_values == val] for val in unique_values}
        total_samples = len(self.data)

        impurity_func = calc_entropy if self.gain_ratio else self.impurity_func # Use entropy if gain_ration = True

        impurity_before = impurity_func(self.data)
        weighted_impurity = sum((len(group) / total_samples) * impurity_func(group) for group in groups.values())

        goodness = impurity_before - weighted_impurity

        if self.gain_ratio:
            p_arr = np.array([len(group) / total_samples for group in groups.values()])
            split_information = -np.sum(p_arr * np.log2(p_arr, where=(p_arr > 0)))
            goodness = goodness / split_information if split_information != 0 else 0

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = self.data[:, -1]

        if self.depth + 1 > self.max_depth or len(np.unique(labels)) == 1 or self.data.shape[1] <= 1:
            self.terminal = True
            return

        best_gain = -1
        best_feature = None
        best_groups = None

        for feature in range(self.data.shape[1] - 1): # all columns except for the labels
            gain, groups = self.goodness_of_split(feature)
            if best_gain < gain:
                best_gain = gain
                best_feature = feature
                best_groups = groups

        if best_gain <= 0 or best_groups is None:
            self.terminal = True
            return

        self.feature = best_feature

        # ---------- Chi-square pruning ----------
        classes, class_counts = np.unique(labels, return_counts=True)
        total_per_class = dict(zip(classes, class_counts))
        total_samples = len(self.data)

        observed = []
        for group in best_groups.values():
            group_labels = group[:, -1]
            group_counts = dict(zip(*np.unique(group_labels, return_counts=True)))
            observed.append([group_counts.get(c, 0) for c in classes])

        observed = np.array(observed)
        expected = np.zeros_like(observed, dtype=float)

        for i, group in enumerate(best_groups.values()):
            group_size = len(group)
            for j, c in enumerate(classes):
                expected[i, j] = (group_size * total_per_class[c]) / total_samples

        nonzero = expected > 0
        chi_stat = np.sum(((observed - expected) ** 2)[nonzero] / expected[nonzero])

        df = (len(classes) - 1) * (len(best_groups) - 1)

        if df in chi_table and self.chi in chi_table[df]:
            threshold = chi_table[df][self.chi]
            if chi_stat < threshold:
                self.terminal = True
                return
        # ---------- End chi-square pruning ----------

        for feature_value, subset in best_groups.items():
            child_node = DecisionNode(data=subset, impurity_func=self.impurity_func, feature=-1,depth=self.depth + 1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio, n_total_sample=self.n_total_sample)
            self.add_child(child_node, feature_value)

        self.calc_feature_importance(n_total_sample=self.n_total_sample)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi,
                                 max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        # Build tree recursively
        def build_tree_recursive(node):
            if not node.terminal:
                node.split()
                for child in node.children:
                    build_tree_recursive(child)

        build_tree_recursive(self.root)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        while not node.terminal:
            feature_val = instance[node.feature]

            # Try to find the child that matches this feature value
            found = False
            for i, val in enumerate(node.children_values):
                if feature_val == val:
                    node = node.children[i]
                    found = True
                    break

            if not found:
                # No matching child found; return majority class of current node
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        predictions = np.array([self.predict(row) for row in dataset])
        true_labels = dataset[:, -1]
        accuracy = np.mean(predictions == true_labels) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

    def depth(self):
        return self.root.depth


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            max_depth=max_depth,
            gain_ratio=True
        )
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]

    for chi in chi_values:
        tree = DecisionTree(
            data=X_train,
            impurity_func=calc_entropy,
            chi=chi,
            max_depth=1000,
            gain_ratio=True
        )
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(tree.depth())  # assumes DecisionNode.depth() works recursively

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node.terminal:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # return n_nodes








