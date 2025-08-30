"""
Module: ngram_featurizer
Author: Peyton Taylor
Description:
    Defines the public API for creating unigram, bigram, and trigram features
    from tokenized text.

    Includes functions for tokenizing text based on white space and creating n-gram features.

    Intended as a reusable library module.

Functions:
   - tokenize_whitespace(text: str) -> list[str]
   - featurize_unigram(tokens: list[str]) -> dict[str, int]
   - featurize_bigram(tokens: list[str]) -> dict[str, int]
   - featurize_trigram(tokens: list[str]) -> dict[str, int]

Function Examples:
    >>> text = "This is an example"
    >>> tokens = tokenize_whitespace(text)
    ['This', 'is', 'an', 'example']
    >>> featurize_bigram(tokens)
    {'This is': 1, 'is an': 1, 'an example': 1}
"""

def tokenize_whitespace(text: str) -> list[str]:
    """
    Very Simple tokenizer. Tokenize by splitting on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str]: A list of tokens.
    """
    tokens = text.split(" ")
    
    return tokens


def featurize_unigram(tokens: list[str]) -> dict[str, int]:
    """
    Creates a dictionary of unigram counts from a list of tokens.
    
    Args:
        tokens (list[str]): A list of tokenized words.

    Returns:
        dict[tuple[str], int]: Dictionary mapping unigram tuples to their frequency.
    """
    unigram_feats = {}
    for i in range(len(tokens) - 1):
        unigram = (tokens[i],) # 1 item tuple for ngram type consistency
        if unigram not in unigram_feats:
            unigram_feats[unigram] = 1
        else:
            unigram_feats[unigram] += 1

    return unigram_feats


def featurize_bigram(tokens: list[str]) -> dict[tuple[str, str], int]:
    """
    Creates a dictionary of bigram counts from a list of tokens.
    
    Args:
        tokens (list[str]): A list of tokenized words.

    Returns:
        dict[tuple[str, str], int]: Dictionary mapping bigram tuples to their frequency.
    """
    bigram_feats = {}
    for i in range(len(tokens) - 1):
        bigram = tuple(tokens[i:i+2])
        if bigram not in bigram_feats:
            bigram_feats[bigram] = 1
        else:
            bigram_feats[bigram] += 1

    return bigram_feats


def featurize_trigram(tokens: list[str]) -> dict[tuple[str, str, str], int]:
    """
    Creates a dictionary of trigram counts from a list of tokens.
    
    Args:
        tokens (list[str]): A list of tokenized words.

    Returns:
        dict[tuple[str, str, str], int]: Dictionary mapping trigram tuples to their frequency.
    """
    trigram_feats = {}
    for i in range(len(tokens) - 1):
        trigram = tuple(tokens[i:i+3])
        if trigram not in trigram_feats:
            trigram_feats[trigram] = 1
        else:
            trigram_feats[trigram] += 1

    return trigram_feats