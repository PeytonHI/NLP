Author: Peyton Taylor
Natural Language Processing Course @ UH Hilo

NLP Projects – UH Hilo

Projects completed as part of the Natural Language Processing course under Professor Winston Wu.

# Overview:
This repository contains two main implementations:

N-gram Language Model
Naive Bayes Text Classifier

Both projects were developed in Python and include preprocessed datasets for testing and evaluation.
Datasets are included in the data/ directory.

# Features:
N-gram Language Model:
Supports unigram, bigram, and trigram models
Implements smoothing techniques
Calculates perplexity on training and dev sets

Naive Bayes Classifier:
Handles text classification using tokenized input
Outputs accuracy metrics on provided datasets

# How to Run:
Clone the repository or download the ZIP file:

git clone https://github.com/yourusername/nlp.git
cd nlp

# Run the desired script:
Language Model:
python run_lm.py

Naive Bayes Classifier:
python run_naivebayes.py

# Requirements:
Python 3.10.16

# Required libraries (can install with pip install -r requirements.txt)::
tokenizers

# Repository Structure:
├─ nlp/
├─ ..
│
├─  ngram_lm_nb
├─ run_lm.py 
├─ run_naivebayes.py
└─ data/
