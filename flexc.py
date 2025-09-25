# Copyright (c) 2025 Mattia Fanan
# Copyright (c) 2025 Iclem Naz Bakirci
# Copyright (c) 2025 Ruggero Carli
# Copyright (c) 2025 Gian Antonio Susto
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# File name: flexc.py
# Project: FLEX-C
# Date (format yyyy-mm-dd): 2025-09-25
# Authors: Mattia Fanan, Iclem Naz Bakirci, Ruggero Carli, Gian Antonio Susto
# Description: This file contains FLEX-C Labels and FLEX-C Centroids implementations of "Forest of Local EXperts Classifiers" (FLEX-C) framework. 


from sklearn.ensemble import IsolationForest
import numpy as np
from scipy import stats
from sklearn_mod_functions import decision_function_single_tree, _score_samples


class UnlabelledAnomalyError(Exception):
    def __init__(self, message):
        super().__init__(message)

class LeafInfo:
    def __init__(self):
        self.label_ = None
        self.centroid_ = None
        self.depth = None

    def update_label(self, labels: np.ndarray):
        mode = stats.mode(labels, axis=0, nan_policy='omit').mode
        self.label_ = mode

    def update_depth(self, depth: int):
        self.depth = depth

    def update_centroid(self, points: np.ndarray):
        if self.centroid_ is None:
            self.n_ = points.shape[0]
            self.centroid_ = points.mean(axis=0)
        else:
            self.centroid_ = (self.n_ * self.centroid_ + points.sum(axis=0)) / (self.n_ + points.shape[0])
            self.n_ += points.shape[0]

    def label(self):
        return self.label_

    def centroid(self):
        return self.centroid_
    
class LeafInfoKNN:
    def __init__(self):
        self.data = {}
        self.depth = None

    def update(self, x: np.ndarray, y: np.ndarray):
        # Vectorized aggregation: sum and count per label
        labels, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        sums = np.zeros((len(labels), x.shape[1]))
        np.add.at(sums, inverse, x)

        for i, k in enumerate(labels):
            new_sum = sums[i]
            new_count = counts[i]
            if k not in self.data:
                self.data[k] = (0, None)
            n, centroid = self.data[k]
            if centroid is None:
                centroid = new_sum / new_count
            else:
                centroid = (n * centroid + new_sum) / (n + new_count)
            self.data[k] = (n + new_count, centroid)


    def set_depth(self, depth: int):
        self.depth = depth

    def get_centroids(self):
        """
        Returns:
            - labels: unique labels in the leaf
            - centroids: centroids for each label
        """
        if not self.data:
            return [], []

        keys, values = zip(*self.data.items())
        centroids = np.array([v[1] for v in values])
        return np.array(keys), centroids
    
    def predict(self, x: np.ndarray):
        """
        Predict labels for one or more input vectors x.
        Returns:
            - predicted labels (closest centroid)
            - full distance matrix: shape (n_samples, n_classes)
        """
        if not self.data:
            return np.array([]), np.array([])

        keys, centroids = self.get_centroids()

        x = np.atleast_2d(x)

        x_norm = (x ** 2).sum(axis=1, keepdims=True)
        c_norm = (centroids ** 2).sum(axis=1)
        dot = x @ centroids.T
        dists = np.sqrt(x_norm + c_norm - 2 * dot) # shape (n_samples, n_labels)
        return dists, keys



class FLEXC_Labels(IsolationForest):
    def __init__(
            self,
            *,
            n_estimators=100,
            max_samples="auto",
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

    def fit(self, X, y=None, sample_weight=None):
        """
        Unsupervisd Isolation Forest structure initializer
        """

        if type(X) is not np.ndarray:
            raise TypeError("X must be a numpy array")
        super().fit(X, y, sample_weight)
        # initialize all nodes' info to an empty LeafInfo
        self.leafs_info_ = []
        for estimator in self.estimators_:
            self.leafs_info_.append(np.array([LeafInfo() for _ in range(estimator.tree_.node_count)], dtype=object))
        # assign leafs' centroids
        for i, estimator in enumerate(self.estimators_):
            leafs = estimator.apply(X)
            for leaf in np.unique(leafs):
                self.leafs_info_[i][leaf].update_centroid(X[leafs == leaf])
        # assign leafs' depths
        for i, estimator in enumerate(self.estimators_):
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            stack = [(0, 0)]
            while len(stack) > 0:
                node_id, depth = stack.pop()
                if children_left[node_id] == -1:
                    self.leafs_info_[i][node_id].update_depth(depth)
                else:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
        return self

    def predict_labels(self, X):
        if type(X) is not np.ndarray:
            raise TypeError("X must be a numpy array")
        # TODO maybe we can also update the centroids here
        # Get the leaf labels for the training data for each tree
        leaves_all_trees = np.column_stack(
            [
                np.array([leaf.label() for leaf in self.leafs_info_[i][estimator.apply(X)]])
                for i, estimator in enumerate(self.estimators_)
            ]
        )

        # Find the most recurrent label across all trees
        mode_result = stats.mode(leaves_all_trees.astype(np.float32), axis=1, nan_policy='propagate')
        return mode_result.mode, mode_result.count/leaves_all_trees.shape[1]

    def predict_one_label(self, X, alarm_threshold=0.0):
        if type(X) is not np.ndarray:
            raise TypeError("X must be a numpy array")
        # TODO maybe we can also update the centroids here
        # Get the leaf labels for the training data for each tree
        leaves_all_trees = [
                np.array([leaf.label() for leaf in self.leafs_info_[i][estimator.apply([X])]])
                for i, estimator in enumerate(self.estimators_)
            ]
        leaves_all_trees = np.column_stack(leaves_all_trees)
        # Find the most recurrent label across all trees
        mode_result = stats.mode(leaves_all_trees.astype(np.float32), axis=1, nan_policy='propagate')
        mode = mode_result.mode
        if mode == -1 or mode is None or np.isnan(mode):
            score = self.decision_function([X])
            if score < alarm_threshold:
                raise UnlabelledAnomalyError("Anomaly detected but not labelled")
        return mode

    def inject_knowledge(self, X, y):
        """
        Trains the local classifiers trough supervised samples
        """
        if type(X) is not np.ndarray or type(y) is not np.ndarray:
            raise TypeError("X and y must be numpy arrays")
        for i, estimator in enumerate(self.estimators_):
            # remove points that are not anomalies for this tree
            x_score = decision_function_single_tree(self, i, X)
            X_anomalies = X[x_score < 0]
            x_leafs = estimator.tree_.apply(X)
            for leaf in np.unique(x_leafs):
                points_on_leaf_ind = np.where(x_leafs == leaf)[0]
                self.leafs_info_[i][leaf].update_label(y[points_on_leaf_ind])
        return self
    
def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    if x.size == 0:
        return np.array([])
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def scale_prob(prob, pivot):
    #1/(1+e^(-5(prob-pivot)​))
    return 1 / (1 + np.exp(-5 * (prob - pivot)))


class FLEXC_Centroids(IsolationForest):
    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

    def fit(self, X, y=None, sample_weight=None):
        """
        Unsupervisd Isolation Forest structure initializer
        """
        super().fit(X, y, sample_weight)

        self.leafs_info_ = []
        for estimator in self.estimators_:
            node_count = estimator.tree_.node_count
            leaf_info = np.empty(node_count, dtype=object)
            leaf_info[:] = [LeafInfoKNN() for _ in range(node_count)]
            self.leafs_info_.append(leaf_info)

        # Assign depths using a shared helper function (faster than stack per tree)
        for i, estimator in enumerate(self.estimators_):
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            leaf_info = self.leafs_info_[i]

            stack = [(0, 0)]
            while stack:
                node_id, depth = stack.pop()
                if children_left[node_id] == -1:
                    leaf_info[node_id].set_depth(depth)
                else:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1)) 

        return self

    def predict_labels(self, X: np.ndarray, alarm_threshold=0.0, y=None, alarm_norm=True):
        """
        Calculates a full confidence distribution for each sample in X.

        For each sample, it aggregates the probability distributions from all trees.
        The final confidence for a class is the average probability assigned to it
        across all trees.

        Returns:
            - final_confidences (np.ndarray): Shape (n_samples, n_classes). The probability
                                              distribution over all classes for each sample.
                                              The final prediction can be derived outside by taking:
                                              `self.classes_[np.argmax(final_confidences, axis=1)]`.
            - alarm (np.ndarray): Shape (n_samples,). Boolean flag indicating if a sample
                                  was anomalous and could not be classified by any tree.
        """
        n_samples = X.shape[0]

        # PREREQUISITE: self.classes_ must be defined during `fit()`.
        # It's a sorted array of all unique class labels.
        if not hasattr(self, 'classes_'):
            raise RuntimeError("Model is not fitted yet or 'self.classes_' is not set.")
        
        n_classes = len(self.classes_) + 1 # +1 for the "background" class "0"
        # Create a mapping from class label to its index for efficient lookups
        class_to_idx = {label: i+1 for i, label in enumerate(self.classes_)}

        # Arrays to store the final results for each sample
        final_confidences = np.zeros((n_samples, n_classes))
        alarm = np.zeros(n_samples)

        # Process one sample at a time
        for i in range(n_samples):
            sample = X[i:i+1]  # Keep as 2D array

            # Store the full probability vectors from each tree for the current sample
            per_tree_prob_pred = []
            per_tree_background_pred = []

            for tree_idx, estimator in enumerate(self.estimators_):
                # 1. Find the leaf for the current sample in the current tree
                leaf_index = estimator.apply(sample)[0]
                leaf_info = self.leafs_info_[tree_idx][leaf_index]

                # 2. Use the leaf's k-NN predictor
                dists, keys = leaf_info.predict(sample)

                # 3. this is the case where no imposed label is found
                # it's a background sample
                if len(dists) == 0 or len(keys) == 0:
                    dec_fun = -_score_samples(self, tree_idx, sample)[0]
                    if dec_fun > -self.offset_ - alarm_threshold:
                        alarm[i] += 1
                    else:
                        # _score_samples is -[ 2^(-depth / c(n))]
                        # so close to 0 for normal
                        # and close to -1 for anomalous
                        # offset is the lowest score for 90 percent of the points
                        # less than offset is anomalous
                        #######
                        # dec fun is opposite of the score
                        # so close to 1 for anomalous
                        # and close to 0 for normal
                        ######
                        # scale_prob puts the 50% confidence at the offset
                        # confidence for sample i being anomalous (so 1-that it is background)
                        per_tree_background_pred.append(1-scale_prob(dec_fun, -self.offset_))
                        #n_background_pred.append(dec_fun)
                    continue

                # 4. Convert distances to a local probability distribution
                scores = -dists[0]  # dists is shape (1, n_leaf_classes)
                local_probabilities = _softmax(scores)

                # 5. Create a full probability vector for this tree, initialized to zeros
                tree_full_probs = np.zeros(n_classes)
                
                # 6. Place local probabilities into the correct positions in the full vector
                for key_idx, key in enumerate(keys):
                    if key in class_to_idx:
                        global_idx = class_to_idx[key]
                        tree_full_probs[global_idx] = local_probabilities[key_idx]
                
                per_tree_prob_pred.append(tree_full_probs)

            if len(per_tree_prob_pred) == 0:
                mean_background_pred = np.mean(per_tree_background_pred)
                final_confidences[i, :] = np.ones(n_classes)
                final_confidences[i, 0] = mean_background_pred
                final_confidences[i, 1:] = (1 - mean_background_pred)/(n_classes - 1)
                continue
            if len(per_tree_background_pred) == 0:
                # 6. Aggregate results from all trees for the current sample
                tree_full_probs = np.array(per_tree_prob_pred)
                avg_probs = np.mean(tree_full_probs, axis=0)
                final_confidences[i, :] = avg_probs
                continue
            
            tree_full_probs = np.array(per_tree_prob_pred)
            tree_full_avg_probs = np.mean(tree_full_probs, axis=0)
            per_tree_background_pred = np.array(per_tree_background_pred)

            complete_background_prob = np.hstack((
                per_tree_background_pred.reshape(-1, 1),
                (1 - per_tree_background_pred.reshape(-1, 1)) * np.repeat(tree_full_avg_probs[1:].reshape(1, -1), len(per_tree_background_pred), axis=0)
            ))
            if alarm_norm:
                complete_prob = np.vstack((
                    complete_background_prob, 
                    tree_full_probs,
                    np.repeat(tree_full_avg_probs.reshape(1, -1), alarm[i], axis=0)
                ))
            else:
                complete_prob = np.vstack((
                    complete_background_prob, 
                    tree_full_probs
                ))
            final_confidences[i, :] = np.mean(complete_prob, axis=0)

        return final_confidences, alarm
    
    def inject_knowledge(self, X, y, score_threshold=0.0):
        """
        Trains the local classifiers trough supervised samples
        """
        if not hasattr(self, 'classes_'):
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.union1d(self.classes_, np.unique(y))

        for i, estimator in enumerate(self.estimators_):
            # 1. Get anomaly scores and filter for anomalous points
            x_score = -_score_samples(self, i, X)
            anom_mask = (x_score > -self.offset_ - score_threshold)

            if not np.any(anom_mask):
                continue

            X_anomalous = X[anom_mask]
            y_anomalous = y[anom_mask]

            # 2. Get leaf assignments for the anomalous points
            # leaf_indices is an array like [3, 1, 4, 1, 5, 3, ...]
            leaf_indices = estimator.tree_.apply(X_anomalous)
            
            # If there are no anomalous points after all, continue
            if leaf_indices.size == 0:
                continue

            # 3. The Vectorization Core: Sort data by leaf index
            # This groups all data points for the same leaf together in memory.
            sorter = np.argsort(leaf_indices)
            
            # Apply the sorter to all relevant arrays
            sorted_leaves = leaf_indices[sorter]
            sorted_X = X_anomalous[sorter]
            sorted_y = y_anomalous[sorter]

            # 4. Find the boundaries of each group of identical leaf indices
            # 'unique_leaves' will be the sorted unique leaf IDs (e.g., [1, 3, 4, 5])
            # 'split_points' will be the indices where each new leaf group starts
            # e.g., if sorted_leaves is [1,1,1, 3,3, 4,4,4,4, 5], 
            # split_points would be [0, 3, 5, 9]
            unique_leaves, split_points = np.unique(sorted_leaves, return_index=True)
            
            # 5. Iterate through the much smaller list of unique leaves and update
            # This loop is fast because it only runs K times (for K unique leaves)
            # and slicing is very cheap.
            for j, leaf_id in enumerate(unique_leaves):
                # Define the slice for the current leaf's data
                start = split_points[j]
                # The end is the start of the next group, or the end of the array
                end = split_points[j+1] if j + 1 < len(split_points) else None
                
                # Slice the pre-sorted data to get the batch for this leaf
                x_for_leaf = sorted_X[start:end]
                y_for_leaf = sorted_y[start:end]

                # Call the already-vectorized update method on the correctly grouped data
                self.leafs_info_[i][leaf_id].update(x=x_for_leaf, y=y_for_leaf)
                
        return self
