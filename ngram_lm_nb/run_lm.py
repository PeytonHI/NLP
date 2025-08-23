"""
Module: run_lm
Author: Peyton Taylor
Description:
    Example script for training and evaluating an N-gram language model.

    This script demonstrates:
        - Reading and tokenizing text datasets.
        - Training an N-gram language model.
        - Generating text with greedy and top-k sampling.
        - Computing perplexity on train and development datasets.

    Intended as a usage example, not a reusable library module.

Usage Example:
    >>> python run_lm.py
"""

# Standard library imports
from pathlib import Path

# Local imports
from lm import tokenize_bpe, NGramLanguageModel

def main():
    # Set up file paths
    base_dir = Path(__file__).parent
    sherlock_file = base_dir / "data" / "SH-TTC" / "sherlock.txt"
    ttc_file = base_dir / "data" / "SH-TTC" / "ttc.txt"

    # Read text files
    with open(sherlock_file, encoding ="utf-8") as f_sherlock:
        sherlock_text = f_sherlock.read()
    with open(ttc_file,  encoding ="utf-8") as f_ttc:
        ttc_text = f_ttc.read()

    # Tokenize
    sherlock_tokens = tokenize_bpe(sherlock_text, sherlock_file)
    ttc_tokens = tokenize_bpe(ttc_text, ttc_file)

    # Create N-gram size (adjustable)
    n_gram_size = 4 

    # Train language model
    lm = NGramLanguageModel(n_gram_size) 
    lm.train(sherlock_tokens)

    # Compute token probabilities
    lm.calculate_token_probs(sherlock_tokens)

    # Generate tokens based on greedy sampling 
    start_context = ["dog"]
    sentence_length = 8
    greedy_tokens = lm.generate_greedy(start_context, sentence_length)
    print(f"Greedy tokens: {greedy_tokens} with desired length of {sentence_length}, start context of {start_context}, and ngram size of {n_gram_size}")

    # Generate tokens based on top-k sampling
    top_k_tokens = lm.generate_topk(start_context, sentence_length, k=5)
    print(f"Top-k tokens: {top_k_tokens} with desired length of {sentence_length}, start context of {start_context}, ngram size of {n_gram_size}, and k of 5")

    # Display perplexity
    print("Train perplexity:", lm.calculate_perplexity(sherlock_tokens))
    print("Dev perplexity:", lm.calculate_perplexity(ttc_tokens))


if __name__ == "__main__":
    main()
