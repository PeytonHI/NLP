"""
Module: metrics
Author: Peyton Taylor
Description:
    Provides evaluation metrics for n-gram language models.

    Includes functions to compute precision, recall, and F1 score for predicted labels.

    Intended as a reusable library module.

Functions:
    - precision(truth: list, predictions: list, positive_label: str) -> float:
        Computes the precision metric.

    - recall(truth: list, predictions: list, positive_label: str) -> float:
        Computes the recall metric.

    - f1_score(truth: list, predictions: list) -> float:
        Computes the F1 score metric.

Function Examples:
    >>> truth = ["SH", "TTC", "SH"]
    >>> predictions = ["SH", "SH", "SH"]
    >>> precision(truth, predictions, "SH")
    0.67
    >>> recall(truth, predictions, "SH")
    1
    >>> f1_score(truth, predictions)
    0.8
"""

def precision(truth: list, predictions: list, positive_label: str) -> float:
    """
    Computes the precision metric.

    Args:
        truth (list): The ground truth labels.
        predictions (list): The predicted labels.
        positive_label (str): The label considered as positive for metric calculations.

    Returns:
        float: The precision score.
    """
    true_positive_count = 0
    false_positive_count = 0

    for t, p in zip(truth, predictions):
        if t == positive_label and p == positive_label: # gold label equals predicted label
            true_positive_count += 1
        else:
            if t != positive_label and p == positive_label: # gold label is not positive_label and predicted label is positive_label
                false_positive_count += 1

    precision = true_positive_count / (false_positive_count + true_positive_count) # tp / (tp + fp)

    return precision


def recall(truth: list, predictions: list, positive_label: str) -> float:
    """
    Computes the recall metric.

    Args:
        truth (list): The ground truth labels.
        predictions (list): The predicted labels.
        positive_label (str): The label considered as positive for metric calculations.

    Returns:
        float: The recall score.
    """
    true_positive_count = 0
    false_negative_count = 0

    for t, p in zip(truth, predictions):
        if t == positive_label and p == positive_label: # gold label equals predicted label
            true_positive_count += 1

        else:
            if t == positive_label and p != positive_label: # gold label is positive_label and predicted label is not positive_label
                false_negative_count += 1

    recall = true_positive_count / (true_positive_count + false_negative_count) # tp / (tp + fn)

    return recall


def f1_score(truth: list, predictions: list, positive_label: str) -> float:
    """
    Computes the f1 metric.

    Args:
        truth (list): The ground truth labels.
        predictions (list): The predicted labels.
        positive_label (str): The label considered as positive for metric calculations.

    Returns:
        float: The F1 score.
    """
    total_precision = precision(truth, predictions, positive_label)
    total_recall = recall(truth, predictions, positive_label)

    f1 = (2*total_precision*total_recall) / (total_precision + total_recall) # (2PR) / (P+R)

    return f1



