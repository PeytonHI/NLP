"""
Module: naive_bayes
Author: Peyton Taylor
Description:
    This module is a Naive Bayes classifier implementation.

    Includes functions for training, predicting, and computing probabilities.

    Intended as a reusable library module, not an example script.

Classes:
    - NaiveBayesClassifier
        A naive bayes classifier with methods for training and predicting.

Functions:
    - train(self, X: list[dict[tuple[str, ...], int]], y: list[str])
        Train the classifier with labeled data.

    - prior(self, y)
        Compute the prior probability of a label.

    - likelihood(self, x, y)
        Compute the likelihood of a feature given a label.

    - predict(self, x: dict)
        Predict the most likely label for a given input.

Function Examples:
    >>> classifier = NaiveBayesClassifier()
    >>> classifier.train(X_train, y_train)
    >>> classifier.prior("some_label")
    -1.4
    >>> classifier.likelihood("some_feature", "some_label")
    -2.3
    >>> classifier.predict(X_test)
    "SH"
"""

# Standard library imports
from collections import defaultdict
from math import log

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for text classification.

    Attributes:
        trained (bool): Indicates whether the model has been trained.
        label_count (defaultdict[int]): Counts of each label in the training data.
        feature_count (dict): Counts of each feature given a label.
        total_sentences (int): Total number of training instances.
    """
    def __init__(self):
        self.trained = False
        self.label_count = defaultdict(int)
        self.feature_count = {}
        self.total_sentences = 0

    def train(self, X: list[dict[tuple[str, ...], int]], y: list[str]):
        """
        Learn model parameters p(w | y) and p(y).

        Args:
            X (list[dict[tuple[str, ...], int]]): The input features.
            y (list[str]): The corresponding labels.
        """

        # Example at this stage - initial nested defaultdict: {label: {}} -> {"SH": {}, "TTC": {}, ...}
        for features, label in zip(X, y):
            if label not in self.feature_count: # make new dict if label not in feature_count
                self.feature_count[label] = {}

            # Example at this stage - non-empty nested dict: {label: {(feature): count}}
            # {label: {(feature): count}} -> {"SH": {('“On',): 8, ('entering',): 2, ...}, "TTC": {...}, ...}
            for feature in features:
                if feature not in self.feature_count[label]: # initialize feature count if not present
                    self.feature_count[label][feature] = 0
                self.feature_count[label][feature] += 1

            self.label_count[label] += 1 
            self.total_sentences += 1 # features are sentences

        self.trained = True

    def prior(self, y: str) -> float:
        """
        Return the prior p(y). Probability of choosing random feature with laplace smoothing and it being a label y before considering features (training set only).

        Args: 
            y (str): The label for which to compute the prior probability.

        Returns:
            float: The prior probability of the label.
        """
        
        if not self.trained:
            raise Exception("Must train first!")

        label_count = self.label_count[y]
        total_labels_count = self.total_sentences
        unique_labels_count = len(self.label_count)

        # Log of probabiltiy of prior y equals log(prior[y]) -> log(y labelled sentences + 1 for laplace smoothing, divided by total num of sentences + length of all labels)
        prob_prior_y = log((label_count + 1) / (total_labels_count + unique_labels_count))

        return prob_prior_y
    
    def likelihood(self, x: str | list[str], y: str) -> float:
        """
        Return the likelihood p(x | y). Probability of observing feature(s) x given label y.

        Args:
            x (str | list[str]): The input feature(s) for which to compute the likelihood.
            y (str): The label given to the input feature(s).

        Returns:
            float: The likelihood of the feature(s) given the label.
        """

        if not self.trained:
            raise Exception("Must train first!")

        # Counts of this feature labelled y
        feature_count_y = self.feature_count[y].get(x, 0)

        # Total counts of all features for label y
        total_features_y = sum(self.feature_count[y].values())

        # Vocabulary size for Laplace smoothing
        vocab_size = len(self.feature_count[y])

        # Smoothed likelihood
        likelihood = log((feature_count_y + 1) / (total_features_y + vocab_size))

        return likelihood
    
    def predict(self, x: dict, display: str = "no") -> str:
        """
        Display and Return the most likely label for the given feature dictionary.

        Args:
            x (dict): The feature dictionary for which to predict the label.
        
        Returns:
            str: The predicted label for the given feature dictionary.
        """

        if not self.trained:
            raise Exception("Must train first!")

        # Log space between 0 and 1, always negative so greatest is best
        max_prob = float("-inf") 
        max_label = None

        # For each feature, get the probability of prior label and likelihood of each feature then set new likely label
        for label in self.label_count:
            probability_prior_label = self.prior(label)

            for feature in x.keys():
                likelihood = self.likelihood(feature, label)

                predict_value = probability_prior_label + likelihood

                if predict_value > max_prob:
                    max_prob = predict_value
                    max_label = label

        # Display feats and predicted labels
        if display == "yes" or display == "y":
            print("Features:", list(x.keys()), "Predicted Label:", max_label, "\n")

        return max_label
