import random
from collections import Counter
from math import log, exp
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

class NGramLanguageModel:
    def __init__(self, n: int):
        self.n = n
        self.trained = False
        self.tokens = []
        self.ngram_key = ()
        self.ngram_key_minus1 = ()
        self.ngram_counter = Counter()
        self.ngram_minus1_counter = Counter()
        self.token_probability = {}


    def train(self, tokens: list[str]):
        """
        Learn the parameters of your language model. Input is already tokenized.
        """
        self.tokens = tokens

        # Counters of ((tuple key): value)
        token_length = len(tokens)
        # loop through tokens while n available
        for i in range(token_length - (self.n + 1)):
            # slices of tokens are the keys. 4gram, 3gram, etc.
            self.ngram_key = tuple(tokens[i:i + (self.n)])
            self.ngram_key_minus1 = tuple(tokens[i:i + (self.n-1)])
            # pass in ngram keys into Counter() and count
            self.ngram_counter[self.ngram_key] += 1
            self.ngram_minus1_counter[self.ngram_key_minus1] += 1

        self.trained = True
        

    def prob(self, tokens: list[str]) -> float:
        """
        Compute p(tokens)
        """
        token_count = {}
        for token in tokens:
            if token not in token_count:
                token_count[token] = 0
            else:
                token_count[token] += 1

        for token, count in token_count.items():
            probability = count/len(tokens)
            self.token_probability[token] = probability


    def generate_greedy(self, start_context: list[str], length: int) -> list[str]:
        """
        Generate `length` tokens using greedy sampling
        """
        if not self.trained:
            raise Exception("Must train first!")

        ngram_counter = Counter()
        new_start = []

        for word in start_context:
            word = word.lower()
            new_start.append(word)
        final_tokens = new_start.copy()

        n = len(start_context)
        for _ in range(length-len(new_start)):  # generate bigrams until desired length is reached
            next_likely_token = {}
            for i in range(len(self.tokens) - (n+1)):
                ngram_key = tuple(self.tokens[i: i + 2]) #bigrams
                ngram_counter[ngram_key] += 1

            start_context_tuple = (final_tokens[-1],) # start context is last token of start context
            next_likely_token = { # add last token in ngram if tokens of ngram up to last token but not including, are equal to last token of start context
                ngram[-1]: count for ngram, count in ngram_counter.items() if ngram[:-1] == start_context_tuple
                }
 
            if next_likely_token: # pull max token from next likely token dictionary.
                next_token = max(next_likely_token,
                                key=next_likely_token.get)

            final_tokens.append(next_token)

        return final_tokens
    
        
    def generate_topk(self, start_context: list[str], length: int, k: int) -> list[str]:
        """
        Generate `length` tokens using top-k/nucleus sampling
        """
        if not self.trained:
            raise Exception("Must train first!")

        ngram_counter = Counter()
        new_start = []

        for word in start_context:
            word = word.lower()
            new_start.append(word)
        final_tokens = new_start.copy()

        n = len(start_context)
        for _ in range(length-len(new_start)):  # generate bigrams until desired length is reached
            next_likely_token = {}
            for i in range(len(self.tokens) - (n+1)):
                ngram_key = tuple(self.tokens[i: i + 2]) #bigrams
                ngram_counter[ngram_key] += 1

            start_context_tuple = (final_tokens[-1],) # start context is last token of start context

            next_likely_token = { # add last token in ngram if tokens of ngram up to last token but not including, are equal to last token of start context
                ngram[-1]: count for ngram, count in ngram_counter.items() if ngram[:-1] == start_context_tuple
                }
            next_likely_token_counter = Counter(next_likely_token) # convert to counter to use most_common method
            top_k = next_likely_token_counter.most_common(k) # pull top k as tuple in list

            if top_k: #random choice of top k amount to choose as next likely token
                unpacked_tokens = [token for token, count in top_k] #unpack tuple from list [(token, count)] -> []
                next_token = random.choice(unpacked_tokens)

            final_tokens.append(next_token)

        return final_tokens


    def perplexity(self, tokens: list[str]):
        """
        Calculates perplexity on the given tokens
        """
        if not self.trained:
            raise Exception("Must train first!")

        unique_tokens = set(tokens)
        ngram_vocab_size = len(unique_tokens)
        probability = {}
        log_sum = 0

        # sum count of ngrams, add 1 for laplace smoothing. Divide by ngram count of ngram 1 less size and total ngram count
        for ngram, count in self.ngram_counter.items():
            each_4gram_probability = (
                count + 1) / (self.ngram_minus1_counter.get(self.ngram_key_minus1, 0) + ngram_vocab_size)
            probability[ngram] = each_4gram_probability # update ngram probability of each

        for ngram, ngram_prob in probability.items():  # sum probability of ngrams with sum of logs
            log_sum += log(ngram_prob)

        token_length = len(tokens)
        perplexity = 0
        perplexity = exp(-log_sum/token_length)  # perplexity formula

        return perplexity


def tokenize(text: str) -> list[str]:
    # text = text.lower()
    # tokens = text.split()
    # return tokens
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer()
    file=["sherlock.txt"]
    tokenizer.train(file, trainer)
    encoded = tokenizer.encode(text)
    tokens = encoded.tokens
    return tokens


def detokenize(tokens: list[str]) -> str:
    # raise NotImplementedError()

    for token in tokens:
        str(token)
