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
"""

# Standard library imports
from pathlib import Path

# Local imports
from lm import tokenize, NGramLanguageModel

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
    sherlock_tokens = tokenize(sherlock_text)
    ttc_tokens = tokenize(ttc_text)

    # Create N-gram size
    n_gram_size = 4 # adjustable n-gram size 

    # Train language model
    lm = NGramLanguageModel(n_gram_size) 
    lm.train(sherlock_tokens)

    # Compute token prediction probabilities
    lm.prob(sherlock_tokens)

    # Generate tokens based on greedy sampling 
    greedy_tokens = lm.generate_greedy(start_context=["dog"], length=8)
    print(greedy_tokens)

    # Generate tokens based on top-k sampling
    top_k_tokens = lm.generate_topk(start_context=["dog"], length=8, k=5)
    print(top_k_tokens)

    # Display perplexity
    print("Train perplexity:", lm.perplexity(sherlock_tokens))
    print("Dev perplexity:", lm.perplexity(ttc_tokens))


if __name__ == "__main__":
    main()
