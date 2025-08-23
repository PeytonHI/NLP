"""
Module: run_naivebayes
Author: Peyton Taylor
Description:
    Example script for training and evaluating a Naive Bayes language model.
    
    This script demonstrates:
        - Reading and processing tab-separated text data.
        - Loading and featurizing datasets.
        - Training and evaluating the model.
        - Using a Naive Bayes classifier for text classification.
    
    Intended as a usage example, not a reusable library module.

Functions:
    - read_data(file_name: str) -> tuple[list[str], list[str]]:
        Reads tab-separated text data from a file.

    - load_train_dataset(train_path: Path) -> tuple[list[str], list[str]]:
        Loads the training dataset from the specified path.

    - load_dev_dataset(dev_path: Path) -> tuple[list[str], list[str]]:
        Loads the development dataset from the specified path.

    - featurize_dataset(dataset: list[str], ngram: str) -> list[dict[tuple[str, ...], int]]:
        Converts text data into feature representations using the specified n-gram type.

    - main():
        Executes the Naive Bayes training and evaluation workflow.

Usage Example:
    >>> python run_naivebayes.py
"""

# Standard library imports
from pathlib import Path

# Local Imports
from naive_bayes import NaiveBayesClassifier
from ngram_featurizer import tokenize_whitespace, featurize_unigram, featurize_bigram, featurize_trigram
import metrics

def read_data(file_name: Path) -> tuple[list[str], list[str]]:
    """
    Reads tab separated text data from a file. Expects format of <label><tab><text>
    Example:
        SH	“Such paper could not be bought under half a crown a packet.

    Args:
        file_name (Path): The path to the input file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists - the texts (multiple strings in list) and the labels.
    """

    with open(file_name, encoding ="utf-8") as fin:
        texts = []
        labels = []
        for line in fin:
            y, x = line.strip().split('\t') # backwards in file
            texts.append(x)
            labels.append(y)

    return (texts, labels)


def load_train_dataset(train_path: Path) -> tuple[list[str], list[str]]:
    """
    Loads training dataset from the specified path.

    Args:
        train_path (Path): The path to the training data file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing the training texts and labels.
    """

    train_x, train_y = read_data(train_path)

    return (train_x, train_y)


def load_dev_dataset(dev_path: Path) -> tuple[list[str], list[str]]:
    """
    Loads development dataset from the specified path.

    Args:
        dev_path (Path): The path to the development data file.

    Returns:
        tuple[list[str], list[str]]: A tuple containing the development texts and labels.
    """

    dev_x, dev_y = read_data(dev_path)

    return (dev_x, dev_y)


def featurize_dataset(dataset: list[str], ngram: str) -> list[dict[tuple[str, ...], int]]:
    """
    Featurizes the dataset using the specified n-gram type.
    Args:
        dataset (list[str]): The input dataset (list of text samples).
        ngram (str): The type of n-gram to use ("unigram", "bigram", "trigram", etc.).

    Returns:
        list[dict[tuple[str, ...], int]]: A list of dictionaries mapping n-gram tuples to their frequency counts.
    """

    if ngram == "unigram":
        return [featurize_unigram(tokenize_whitespace(x)) for x in dataset]
    elif ngram == "bigram":
        return [featurize_bigram(tokenize_whitespace(x)) for x in dataset]
    elif ngram == "trigram":
        return [featurize_trigram(tokenize_whitespace(x)) for x in dataset]
    else:
        raise ValueError("Unsupported ngram type.")


def train_and_evaluate(train_x: list[str], train_y: list[str], dev_x: list[str], dev_y: list[str], positive_label: str):
    """
    Trains and displays the Naive Bayes model Evaluation.

    Args:
        train_x (list[str]): The training feature set.
        train_y (list[str]): The training labels.
        dev_x (list[str]): The development feature set.
        dev_y (list[str]): The development labels.
        positive_label (str): The label considered as positive for metric calculations.
    """
    model = NaiveBayesClassifier()
    model.train(train_x, train_y)

    # Display features and predicted labels or not
    display = input("Display features and predicted labels? Yes or No (huge spam): ").lower()
    while display not in {"yes", "y", "no", "n"}:
        display = input("Invalid input. Please enter Yes or No (huge spam): ").lower()

    print("loading...")
    predictions = [model.predict(x, display) for x in dev_x]

    assert len(dev_y) == len(predictions), "Mismatch in number of predictions and true labels."

    metric_types = ["precision", "recall", "f1_score"]
    
    for metric_name in metric_types:
        assert hasattr(metrics, metric_name), f"{metric_name} metric is not defined"

    for metric_name in metric_types:
        func = getattr(metrics, metric_name)
        print(f"{metric_name.capitalize()}:", func(dev_y, predictions, positive_label))


def main():
    # Set up file paths
    base_dir = Path(__file__).parent
    train_file = base_dir / "data" / "SH-TTC" / "train.tsv"
    dev_file = base_dir / "data" / "SH-TTC" / "dev.tsv"

    # Load datasets
    train_x, train_y = load_train_dataset(train_file)
    dev_x, dev_y = load_dev_dataset(dev_file)

    # N-grams to use
    ngram_names = ["unigram", "bigram", "trigram"]

    # Need a positive label for metrics
    positive_label = "SH"

    # Feature extraction, model training, and evaluation for each n-gram type
    for name in ngram_names:
        print(f"Training with {name} features...\n")
        featurized_train_x = featurize_dataset(train_x, ngram=name)
        featurized_dev_x = featurize_dataset(dev_x, ngram=name)

        print(f"Evaluating for {name} features:\n")
        train_and_evaluate(featurized_train_x, train_y, featurized_dev_x, dev_y, positive_label)
        print()


if __name__ == "__main__":
    main()
