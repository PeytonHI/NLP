"""
Module: lm
Author: Peyton Taylor
Description:
    Defines the public API for training and using an N-gram language model.

    Includes functions for training the model, calculating probability of a sequence, generating texting using greedy sampling and topk sampling, 
    calculating perplexity of a sequence, and tokenizing text based on BPE.

    Intended as a reusable library module.

Classes:
- NGramLanguageModel: A class for training and using n-gram language models.

Functions:
- train: Train the language model on a list of tokens.
- calculate_token_probs: Compute the probability of a sequence of tokens.
- generate_greedy: Generate text using a greedy sampling approach.
- generate_topk: Generate text using top-k sampling.
- calculate_perplexity: Compute the perplexity of a sequence of tokens.
- tokenize_bpe: Tokenize input text into a list of tokens using BPE.

Function Examples:
    - train:
        >>> model = NGramLanguageModel(n=2)
        >>> model.train(tokens)

    - calculate_token_probs:
        >>> prob = model.calculate_token_probs(['Hello', 'world'])
        >>> print(prob)
        

    - generate_greedy:
        >>> generated = model.generate_greedy(['Hello'], length=5)
        >>> print(generated)
        ['dog', 'to', 'help', 'him', 'to', 'do', 'the', 'old']

    - generate_topk:
        >>> topk_generated = model.generate_topk(['Hello'], length=5, k=3)
        >>> print(topk_generated)
        ['dog', 'to', 'help', 'him', 'to', 'do', 'his', 'worst']

    - calculate_perplexity:
        >>> perplexity = model.calculate_perplexity(['Hello', 'world'])
        >>> print(perplexity)
        1.8

    - tokenize_bpe:
        >>> tokens = tokenize_bpe("Hello, world!")
        >>> print(tokens)
        ['Hello', ',', 'world', '!']
"""

# Standard library imports
import random
from collections import Counter
from math import log, exp

# 3rd-party imports
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

class NGramLanguageModel:
    """
    N-gram language model.

    Attributes:
        n_gram_size (int): The size of the n-grams.
        trained (bool): Whether the model has been trained.
        tokens (list[str]): The list of tokens used for training.
        ngram_key (tuple): The current n-gram key.
        ngram_minus1_key (tuple): The current (n-1)-gram key.
        ngram_counter (Counter): The counter for n-grams.
        ngram_minus1_counter (Counter): The counter for (n-1)-grams.
        token_probability (dict): The dictionary of token probabilities.
    """
    def __init__(self, n: int):
        self.n_gram_size = n
        self.trained = False
        self.tokens = []
        self.ngram_counter = Counter()
        self.ngram_key = ()
        self.ngram_minus1_counter = Counter()
        self.ngram_minus1_key = ()
        self.token_probability = {}

    def train(self, tokens: list[str]):
        """
        Learn the parameters of language model.

        Args:
            tokens (list[str]): A list of tokenized text.
        """
        self.tokens = tokens
        token_length = len(tokens)

        # Loop through tokens while n are available
        # ngram_counter looks like: {('the', 'cat'): 5, ('cat', 'sat'): 3, ...}
        # ngram_minus1_counter looks like: {('the',): 5, ('cat',): 3, ...}
        for i in range(token_length - self.n_gram_size + 1):
            self.ngram_key = tuple(tokens[i: i + self.n_gram_size]) # slices of tokens are the keys. 4gram, 3gram, etc.
            self.ngram_minus1_key = tuple(tokens[i: i + (self.n_gram_size - 1)]) # n-gram of 1 less size for perplexity 
            self.ngram_counter[self.ngram_key] += 1
            self.ngram_minus1_counter[self.ngram_minus1_key] += 1

        self.trained = True
        
    def calculate_token_probs(self, tokens: list[str]):
        """
        Compute p(tokens). General probability of a token is a specific token's count / count of all tokens.

        Args:
            tokens (list[str]): A list of tokenized text.
        """
        token_count = {}
        for token in tokens:
            if token not in token_count:
                token_count[token] = 0
            else:
                token_count[token] += 1

        for token, count in token_count.items():
            probability = (count / len(tokens))
            self.token_probability[token] = probability

    def generate_greedy(self, start_context: list[str], sentence_length: int) -> list[str]:
        """
        Compute argmax(P(w | context)).
        Generate length tokens using greedy sampling. Select the token with the highest probability.

        Example: P(ate | the cat) = 0.8 and if ate is the highest probability token, it will be selected.

        Args:
            start_context (list[str]): The initial context to start generation.
            sentence_length (int): The desired length of the sentence after generating token predictions.

        Returns:
            list[str]: The generated sentence as a list of tokens.
        """
        if not self.trained:
            raise Exception("Must train first!")
        
        unigram_counter = Counter()
        for token in self.tokens:
            unigram_counter[token] += 1

        lowercase_start_context = convert_words_lowercase(start_context)  
        final_tokens = lowercase_start_context.copy()
        start_context_len = len(start_context)

        # Generate predictions until desired length is reached
        for _ in range(sentence_length - start_context_len):  
            next_likely_tokens = {}

            # Max context length is capped at ngram size, otherwise adjust for backoff, ensuring ngram sequence always is same length as token_context in
            # this line specifically: if ngram[:-1][-context_length:] == token_context:
            context_length = min(len(final_tokens), self.n_gram_size - 1)

            # Backoff: reduce context size until match is found
            while context_length > 0:
                token_context = tuple(final_tokens[-context_length:])
                for ngram, count in self.ngram_counter.items():
                    if ngram[:-1][-context_length:] == token_context: # Example: ('the', 'cat')[-1] == 'cat' -> this captures the last word in the ngram which is our prediction
                        next_likely_tokens[ngram[-1]] = count
                if next_likely_tokens:
                    break  # Found matches, stop backoff
                context_length -= 1
                
            if not next_likely_tokens:
                most_common_token = max(unigram_counter, key=unigram_counter.get)
                final_tokens.append(most_common_token)
                continue

            # Pull max token from next likely tokens dictionary.
            max_token = None
            for token in next_likely_tokens:
                if max_token is None or next_likely_tokens[token] > next_likely_tokens[max_token]:
                    max_token = token

            final_tokens.append(max_token)

        return final_tokens
    
        
    def generate_topk(self, start_context: list[str], sentence_length: int, k: int) -> list[str]:
        """
        This method generates text by sampling from the top-k most likely next tokens at each step using top-k/nucleus sampling.

        Args:
            start_context (list[str]): The initial context to start generation.
            length (int): The desired length of the generated sequence.
            k (int): The number of top tokens to sample from.

        Returns:
            list[str]: The generated sentence as a list of tokens.
        """
        if not self.trained:
            raise Exception("Must train first!")

        unigram_counter = Counter()
        for token in self.tokens:
            unigram_counter[token] += 1

        lowercase_start_context = convert_words_lowercase(start_context)  
        final_tokens = lowercase_start_context.copy()
        start_context_len = len(start_context)

        # Generate predictions until desired length is reached
        for _ in range(sentence_length - start_context_len):  
            next_likely_tokens = {}

            # Max context length is capped at ngram size, otherwise adjust for backoff.
            # Ensures our context is always the same length as the n-gram even if it's bigger than the n-gram, otherwise adjusts for being smaller than the n-gram
            # Example: if ngram is ('the', 'cat', 'sat') and context_length is 2, token_context will be ('cat', 'sat')
            # otherwise, if context_length is 4, token_context will be ('the', 'cat', 'sat')
            # ensuring ngram sequence always is same length as token_context for this line specifically to work: if ngram[:-1][-context_length:] == token_context:.
            # Example: n_gram_size = 4 -> ("the", "cat", "sat", "on")
            # final_tokens = ["cat"]
            # Without min(len(final_tokens), n_gram_size-1), you might set context_length = 3
            # so token_context = ("cat",) (length 1)
            # and ngram[:-1][-3:] = ("the","cat","sat") (length 3), these will never be equal in this case!!
            context_length = min(len(final_tokens), self.n_gram_size - 1)

            # Backoff: reduce context size until match is found
            while context_length > 0:
                token_context = tuple(final_tokens[-context_length:]) # last context length tokens from final_tokens
                for ngram, count in self.ngram_counter.items():
                    if ngram[:-1][-context_length:] == token_context: # Example: ('the', 'cat')[-1] == 'cat' -> this captures the last word in the ngram which is our prediction
                        next_likely_tokens[ngram[-1]] = count
                if next_likely_tokens:
                    break  # Found matches, stop backoff
                context_length -= 1
                
            if not next_likely_tokens:
                most_common_token = max(unigram_counter, key=unigram_counter.get)
                final_tokens.append(most_common_token)
                continue

            # Pull max token from next likely tokens dictionary.
            max_token = None
            for token in next_likely_tokens:
                if max_token is None or next_likely_tokens[token] > next_likely_tokens[max_token]:
                    max_token = token

            next_likely_token_counter = Counter(next_likely_tokens) # convert to counter to use most_common method
            top_k = next_likely_token_counter.most_common(k) # pull top k as tuple in a list

            if top_k: 
                unpacked_tokens = [token for token, count in top_k] # unpack tuple from list [(token, count)] -> []
                next_token = random.choice(unpacked_tokens) # random choice of top k amount to choose as next likely token
                final_tokens.append(next_token)

        return final_tokens


    def calculate_perplexity(self, tokens: list[str]):
        """
        Calculates perplexity on the given tokens.

        Args:
            tokens (list[str]): A list of tokens to calculate perplexity for.

        Returns:
            float: The calculated perplexity.
        """
        if not self.trained:
            raise Exception("Must train first!")

        vocab = set(tokens)
        vocab_size = len(vocab)
        log_sum = 0

        # Sum count of ngrams, add 1 for laplace smoothing. Divide by ngram count of ngrams 1 less in size + total ngram count for conditional probability
        for i in range(vocab_size - self.n_gram_size + 1):
            ngram = tuple(tokens[i: i + self.n_gram_size])
            ngram_minus1 = ngram[:-1]
            count_ngram = self.ngram_counter.get(ngram, 0)
            count_ngram_minus1 = self.ngram_minus1_counter.get(ngram_minus1, 0)

            prob = (count_ngram + 1) / (count_ngram_minus1 + vocab_size)
            log_sum += log(prob)

        token_length = len(tokens)
        perplexity = exp(-log_sum/token_length) # perplexity formula

        return perplexity


def convert_words_lowercase(words: list[str]) -> list[str]:
    """
    Helper Function to convert all words in the word list to lowercase.

    Args:
        words (list[str]): A list of words.

    Returns:
        list[str]: A new list with all words converted to lowercase.
    """
    return [word.lower() for word in words]


def tokenize_bpe(text: str, file) -> list[str]:
    """
    Tokenizes the input text using Byte Pair Encoding (BPE).

    Args:
        text (str): The input text to tokenize.
        file (Path): The file path for training the tokenizer.

    Returns:
        list[str]: A list of BPE tokens.
    """
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer()
    tokenizer.train([str(file)], trainer) # expects list of file paths
    encoded = tokenizer.encode(text)
    tokens = encoded.tokens

    return tokens