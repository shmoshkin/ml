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

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

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
        unique_labels, counts = np.unique(labels, return_counts=True)
        pred = unique_labels[np.argmax(counts)]
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
        if self.feature == -1:  # Root node or leaf node
            self.feature_importance = 0
            return

        # Calculate parent impurity
        parent_impurity = self.impurity_func(self.data)

        # Calculate weighted impurity of children
        weighted_child_impurity = 0
        for child in self.children:
            weighted_child_impurity += (len(child.data) / n_total_sample) * self.impurity_func(child.data)

        # Calculate feature importance
        self.feature_importance = (len(self.data) / n_total_sample) * (parent_impurity - weighted_child_impurity)
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
        # Get unique values of the feature
        feature_values = np.unique(self.data[:, feature])

        # Calculate parent impurity
        parent_impurity = self.impurity_func(self.data)

        # Split data according to feature values
        for value in feature_values:
            mask = self.data[:, feature] == value
            groups[value] = self.data[mask]

        # Calculate weighted impurity of children
        weighted_child_impurity = 0
        for value, subset in groups.items():
            weighted_child_impurity += (len(subset) / len(self.data)) * self.impurity_func(subset)

        # Calculate goodness of split
        goodness = parent_impurity - weighted_child_impurity

        # If gain ratio is requested, calculate split information
        if self.gain_ratio:
            split_info = 0
            for value, subset in groups.items():
                p = len(subset) / len(self.data)
                split_info -= p * np.log2(p)

            # Avoid division by zero
            if split_info > 0:
                goodness = goodness / split_info
            else:
                goodness = 0
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
        # Check if we should stop splitting
        if self.depth >= self.max_depth:
            self.terminal = True
            return

        # Check if all labels are the same
        if len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        # Find best feature to split on
        best_goodness = -float('inf')
        best_feature = -1
        best_groups = {}

        for feature in range(self.data.shape[1] - 1):  # Exclude label column
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        # If no good split found
        if best_goodness <= 0:
            self.terminal = True
            return

        # Check chi-square pruning
        if self.chi < 1:
            # Calculate chi-square statistic
            chi_stat = 0
            parent_labels, parent_counts = np.unique(self.data[:, -1], return_counts=True)
            parent_probs = parent_counts / len(self.data)

            for value, subset in best_groups.items():
                subset_labels, subset_counts = np.unique(subset[:, -1], return_counts=True)
                expected_counts = parent_probs * len(subset)

                for label, count in zip(subset_labels, subset_counts):
                    idx = np.where(parent_labels == label)[0][0]
                    expected = expected_counts[idx]
                    chi_stat += ((count - expected) ** 2) / expected

            # Get degrees of freedom
            df = (len(parent_labels) - 1) * (len(best_groups) - 1)

            # Check if chi-square value exceeds threshold
            if chi_stat < chi_table[df][self.chi]:
                self.terminal = True
                return

        # Set the best feature
        self.feature = best_feature

        # Create child nodes
        for value, subset in best_groups.items():
            child = DecisionNode(subset, self.impurity_func, depth=self.depth + 1,
                                 chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child, value)

        # Calculate feature importance
        self.calc_feature_importance(len(self.data))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class DecisionTree: # test
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
        # Create root node
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
            feature_value = instance[node.feature]
            # Find the child node with matching feature value
            child_idx = None
            for i, val in enumerate(node.children_values):
                if val == feature_value:
                    child_idx = i
                    break
            if child_idx is None:  # If feature value not found in children
                break
            node = node.children[child_idx]
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
        correct_predictions = 0
        total_samples = len(dataset)

        for instance in dataset:
            pred = self.predict(instance)
            if pred == instance[-1]:  # Compare prediction with actual label
                correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

    def depth(self):
        """
        Calculate the maximum depth of the tree.

        Returns:
        - depth: the maximum depth of the tree
        """

        def get_max_depth(node):
            if node.terminal:
                return node.depth
            return max(get_max_depth(child) for child in node.children)

        return get_max_depth(self.root)


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
            tree = DecisionTree(X_train, calc_entropy, max_depth=max_depth, gain_ratio=True)
            tree.build_tree()
            
            # Calculate accuracies
            train_acc = tree.calc_accuracy(X_train)
            val_acc = tree.calc_accuracy(X_validation)
            
            # Store accuracies
            training.append(train_acc)
            validation.append(val_acc)
            
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
    # Test different chi values (p-value cut-offs)
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    
    for chi in chi_values:
        # Create and build tree with current chi value
        tree = DecisionTree(X_train, calc_entropy, chi=chi, gain_ratio=True)
        tree.build_tree()
        
        # Calculate accuracies and depth
        train_acc = tree.calc_accuracy(X_train)
        val_acc = tree.calc_accuracy(X_test)
        tree_depth = tree.depth()
        
        # Store results
        chi_training_acc.append(train_acc)
        chi_validation_acc.append(val_acc)
        depth.append(tree_depth)
        
    # Plot results
    plt.figure(figsize=(12, 8))
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
    n_nodes = 1 
    for child in node.children_values():
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






