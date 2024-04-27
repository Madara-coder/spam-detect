import numpy as np
import math
import pandas as pd
#TFIDF vectorizer class
class TFIDFVectorizer:
    def __init__(self, max_features=None):
        self.idf = {}
        self.vocab = set()
        self.max_features = max_features

    def fit(self, documents):
        n_docs = len(documents)
        term_counts = {}

        # Count term occurrences in each document
        for document in documents:
            term_seen = set()
            for term in document.split():
                if term not in term_seen:
                    term_seen.add(term)
                    term_counts[term] = term_counts.get(term, 0) + 1

        # Calculate IDF
        for term, count in term_counts.items():
            self.idf[term] = math.log(1+n_docs / (count+1))

        # Update vocabulary
        self.vocab.update(term_counts.keys())

    # Converts into the number form
    def transform(self, documents):
        tfidf_matrix = np.zeros((len(documents), len(self.vocab)))
        for idx, document in enumerate(documents):
            # Calculate TF
            term_freq = {}
            for term in document.split():
                term_freq[term] = term_freq.get(term, 0) + 1

            # Calculate TF-IDF
            for term, freq in term_freq.items():
                if term in self.vocab:
                    term_idx = list(self.vocab).index(term)
                    tfidf_matrix[idx, term_idx] = freq * self.idf[term]
        return np.array(tfidf_matrix)

#Multinominal Naive Bayes Algorithm class
class MultinomialNB1:
    def __init__(self):
        self.class_prior = {}
        self.word_probs = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Compute class prior probabilities
        for cls in self.classes:
            self.class_prior[cls] = np.sum(y == cls) / n_samples

        # Compute conditional probabilities of each word given class
        for cls in self.classes:
            # Select rows where class label is equal to cls
            X_cls = X[y == cls]
            # Calculate total count of words for this class
            total_count = np.sum(X_cls)
            # Calculate total count of each word for this class
            word_counts = np.sum(X_cls, axis=0)
            # Calculate probability of each word given class
            self.word_probs[cls] = (word_counts + 1) / (total_count + n_features)

    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate log probabilities for each class
            class_probs = []
            for cls in self.classes:
                # Initialize probability with log of prior probability of the class
                prob = np.log(self.class_prior[cls])
                # Calculate log likelihood of features given class
                for word, count in enumerate(x):
                    prob += count * np.log(self.word_probs[cls][word])   
                class_probs.append(prob)
            # Choose the class with the highest probability
            predictions.append(self.classes[np.argmax(class_probs)])
        return np.array(predictions)

#KNN Algorithm class
from collections import Counter
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = []
            for i, x_train in enumerate(self.X_train):
                distance = np.sqrt(np.sum((x - x_train) ** 2))
                distances.append((distance, i))
            distances = sorted(distances)[:self.k]
            neighbors = [self.y_train[idx] for _, idx in distances]
            most_common = Counter(neighbors).most_common(1)
            predictions.append(most_common[0][0])
        return predictions

#Logistic Regression Algorithm class
class LogisticRegression1:
  def __init__(self, learning_rate=0.01, num_iterations=100):
    self.learning_rate = learning_rate
    self.num_iterations = num_iterations
    self.weights = None

  def sigmoid(self, z):

    return 1 / (1 + np.exp(-z))

  def fit(self, X, y):
    # Add a bias term to X
    X = np.c_[np.ones(len(X)), X]

    # Initialize weights
    self.weights = np.random.rand(X.shape[1])

    # Training loop
    for _ in range(self.num_iterations):
      # Calculate predicted probabilities
      y_pred = self.sigmoid(X.dot(self.weights))

      # Update weights using gradient descent
      self.weights += self.learning_rate * X.T.dot(y - y_pred)

  def predict(self, X):
    # Add a bias term to X
    X = np.c_[np.ones(len(X)), X]
    return self.sigmoid(X.dot(self.weights))

#Decision Tree Algorithm class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Predicted value for leaf node


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples <= 1:
            return Node(value=self._most_common_label(y))

        # Find the best split
        best_split = self._best_split(X, y)

        if best_split is None:
            return Node(value=self._most_common_label(y))

        feature, threshold = best_split
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):
            thresholds = sorted(set(X[:, feature]))
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold

                # gini-impurity formula
                gini = self._gini_impurity(y[left_indices]) * sum(left_indices) / len(y) + \
                       self._gini_impurity(y[~left_indices]) * sum(~left_indices) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # Gini impurity section.
    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_one(sample, self.tree) for sample in X])

    def _predict_one(self, sample, tree):
        if tree.value is not None:
            return tree.value
        else:
            if sample[tree.feature] < tree.threshold:
                return self._predict_one(sample, tree.left)
            else:
                return self._predict_one(sample, tree.right)