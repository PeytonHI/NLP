"""
Module: lm.py
Author: Peyton Taylor
Description:
    Example script for training and evaluating a BiLSTM language model.

    This script demonstrates:
        - Reading and tokenizing text datasets.
        - Training a BiLSTM language model.
        - Generating text with greedy and top-k sampling.

    Intended as a usage example, not a reusable library module.

Usage Example:
    >>> python lm.py
"""

# Standard library imports
import torch
import json
import random
from collections import Counter
from pathlib import Path

# 3rd party imports
import evaluate
import regex as re
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

# Author: Peyton

class BiLSTMLanguageModel(torch.nn.Module):
    """
    Bi-directional LSTM Language Model

    Attributes:
        emb (torch.nn.Embedding): Embedding layer
        lstm (torch.nn.LSTM): Bi-directional LSTM layer
        linear (torch.nn.Linear): Linear layer
    """
    def __init__(self, voc_size, emb_size, hidden_size):
        super().__init__()
        self.emb = torch.nn.Embedding(voc_size, emb_size)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_size * 2, voc_size)


    def forward(self, input):
        e = self.emb(input)
        # e = e.squeeze()
        output, (hidden, cell) = self.lstm(e)
        # print("Output shape:", output.shape)
        if output.dim() == 2:
            # print("Input to linear layer shape:", output.shape)
            final_output = self.linear(output)
        elif output.dim() == 3:
            last_output = output[:, -1, :]
            # print("Input to linear layer shape:", last_output.shape)
            final_output = self.linear(last_output)  # Use the last LSTM output
        else:
            raise ValueError(f"Unexpected shape: {output.dim()}")
    
        return final_output
    
class FFLLM(torch.nn.Module):
    def __init__(self, voc_size):
        super().__init__()
        self.emb = torch.nn.Embedding(voc_size, 300)
        self.linear1 = torch.nn.Linear(300, 300)
        self.linear2 = torch.nn.Linear(300, voc_size)
        
    def forward(self, x):
        e = self.emb(x)
        e = e.squeeze()  # Your task: figure out why this is necessary
        h = self.linear1(e)
        y = self.linear2(h)
        
        return y #torch.softmax(y, dim=0)

class Vocab:
    def __init__(self, tokens):
        self.vocab = [tok for tok, count in Counter(tokens).most_common()]
        self.tok2idx = {tok: idx + 2 for idx, tok in enumerate(self.vocab)}
        self.tok2idx[0] = "[PAD]"
        self.tok2idx[1] = "[UNK]"
        self.tok2idx[2] = "[<p>]"
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}
    
    def __len__(self):
        return len(self.tok2idx)
    
    def to_idx(self, tok):
        return self.tok2idx.get(tok, 0)

    def to_tok(self, idx):
        return self.idx2tok.get(idx, "[UNK]")
    

# class labelVocab:
#     def __init__(self, labels):
#         self.label_vocab = set(labels)
#         self.label2idy = {label: idy + 2 for idy, label in enumerate(self.label_vocab)}
#         self.label2idy[0] = "[PAD]"
#         self.label2idy[1] = "[UNK]"
#         self.idy2label = {idy: label for label, idy in self.label2idy.items()}

#     def __len__(self):
#             return len(self.label2idy)
    
#     def to_idy(self, label):
#         return self.label2idy.get(label, 0)
    
#     def to_toky(self, idy):
        # return self.idy2label.get(idy, "[UNK]")


class Book(Dataset):
    def __init__(self, tokens, ngrams_list, vocab):
        self.vocab = vocab
        self.data = []
        for gram in ngrams_list:
            x = torch.LongTensor([self.vocab.to_idx(gram[0])])
            y = torch.LongTensor([self.vocab.to_idx(gram[1])])
            y = torch.nn.functional.one_hot(y, len(self.vocab)).float().squeeze(0)
            self.data.append((x, y))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(model, train_data, val_data):
    # raise NotImplementedError()
    # setup the training
    loss_func = torch.nn.BCEWithLogitsLoss()  # Use BCE with logits for multi-label classification
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1):
        print("Epoch", epoch)
        for x, y in tqdm(train_data):
            model.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            total_loss = 0
            for x, y in train_data:
                pred = model(x)
                loss = loss_func(pred, y)
                total_loss += loss
            print("train loss:", total_loss / len(train_data))

            total_loss = 0
            for x, y in val_data:
                pred = model(x)
                loss = loss_func(pred, y)
                total_loss += loss
            print("dev loss:", total_loss / len(val_data))


def run_model_on_dev_data(val_data, model):
    preds = []
    refs = []
    model.eval()
    with torch.no_grad():
        # print(f"Number of batches in val_data: {len(val_data)}")
        for x, y in val_data:
            # print(x.shape)
            # print("x shape in val data", x.shape, "y shape:", y.shape)
            pred = model(x)  # pred is something like [0.6, 0.4]
            # print(pred)
            # TODO: when using batched inputs, your output will also be batched
            # so you need to split them before appending to preds

            # print(f"Predictions shape: {pred.shape}, References shape: {y.shape}")  # For debugging 
            # print(f"Raw prediction: {pred}")

            probabilities = torch.sigmoid(pred)
            # print("Probs:", probabilities[:5])  # Debugging: check first few predicted probabilities

            preds.append((probabilities > 0.2).int())     
            refs.append(y)

    preds = torch.cat(preds, dim=0)  # Shape [total_samples, num_labels]
    refs = torch.cat(refs, dim=0)
    # print(f"Predictions shape: {preds.shape}, References shape: {refs.shape}")  # For debugging 

    return preds, refs


def sample_predictions(preds, dev_data_raw, vocab):
    for _ in range(5):
        pred_labels = []

        idx = random.randint(0, len(dev_data_raw) - 1)
        # text = vocab.to_tok(idx)
        gold_text = dev_data_raw[idx][1]
        # gold_label = label_vocab.idy2label[idx]
        gold_label = dev_data_raw[idx][0]

        for i, pred in enumerate(preds[idx]):
                if pred == 1:
                    label = vocab.idy2label[i]
                    pred_labels.append(label)
        
        print("Input: ", gold_text)
        print("Gold label:", gold_label)
        print("Predicted labels: ", pred_labels)


def build_ngrams(tokens, n):
    return [tokens[i: i + n] for i in range(len(tokens) - n + 1)]


def load_data():
    base_dir = Path(__file__).parent 
    sherlock_data_path = base_dir / "data" / "SH-TTC" / "sherlock.txt"
    sherlock_tokens_list = []
    with open(sherlock_data_path, "r", encoding='UTF-8') as sherlock_file:
        sherlock_text = sherlock_file.read().strip()
        paragraphs = re.split(r'\n{2,}', sherlock_text) # paragraphs are separated by two newlines and we want paragraphs for more context
        for paragraph in paragraphs:
            sherlock_tokens = tokenize(paragraph).strip()
            sherlock_tokens_list.append(sherlock_tokens)

    ttc_tokens_list = []
    ttc_data_path = base_dir / "data" / "SH-TTC" / "ttc.txt"
    with open(ttc_data_path, "r", encoding='UTF-8') as ttc_file:
        ttc_text = ttc_file.read().strip()
        paragraphs = re.split(r'\n{2,}', ttc_text)
        for paragraph in paragraphs:
            ttc_tokens = tokenize(paragraph).strip()
            ttc_tokens_list.append(ttc_tokens)

    return sherlock_tokens_list, ttc_tokens_list


def tokenize(text):
    tokenizer = Tokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    return tokens


def main():

    sherlock_text, ttc_tokens = load_data()
    sherlock_tokens = tokenize(sherlock_text)
    ttc_tokens = tokenize(ttc_tokens)
    # print(train_data_raw[:5])
    # print(val_data_raw[:5])

    vocab = Vocab(sherlock_tokens)
    # label_vocab = labelVocab(sherlock_tokens)
    n = 4
    ngrams_list = build_ngrams(sherlock_tokens, n)
    # split the data into train and dev
    # train_part = int(len(sherlock_tokens) * 0.8)
    train_data = Book(sherlock_tokens, ngrams_list, vocab)
    val_data = Book(ttc_tokens, ngrams_list, vocab)
    
    train_data[0]
    torch.nn.functional.one_hot(torch.LongTensor([5]), 10)
    torch.argmax(train_data[1][1])
    # vocab.to_tok(5167)

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)
    # print(len(train_dataloader))

    # for batch in dataloader:
    #     texts, labels = batch
    #     print(f'Texts: {texts}')
    #     print(f'Labels: {labels}')


    # print("Training dataset size:", len(train_dataset))
    # print("Validation dataset size:", len(val_dataset))

    # # Ensure DataLoader outputs match expected sizes
    # for batch_idx, (inputs, labels) in enumerate(train_dataloader):
    #     print(f"Batch {batch_idx} - Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

    voc_length = len(vocab)
    model = BiLSTMLanguageModel(voc_length, 100, 100)
    
    # model = FFLLM(len(vocab))

    train(model, train_dataloader, val_dataloader)
    
    preds, refs = run_model_on_dev_data(val_dataloader, model)
    preds = preds.int()
    refs = refs.int()

    # print("Final preds shape:", preds.shape)
    # print("Final preds:", preds[:5])  # Check the first 5 predictions
    # print("Final refs shape:", refs.shape) 
    # print("Final refs:", refs[:5])  # Check the first 5 refs
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    accuracy = evaluate.load("accuracy")

    sample_predictions(preds, val_data, label_vocab)

    input(" ------- Press Any key to continue ------- ")

    preds_binary = []
    for pred in preds: 
        for i in pred:  # Iterate through each element in the tensor
                preds_binary.append(i.item())  # Append the value to the list

    refs_binary = []
    for ref in refs:  # Iterate through the tensor `refs`
        for i in ref:
                refs_binary.append(i.item())

    # print("Predictions shape: ", preds.shape, "Refs shape: ", refs.shape)
    print(precision.compute(references=refs_binary, predictions=preds_binary))
    print(recall.compute(references=refs_binary, predictions=preds_binary))
    print(accuracy.compute(references=refs_binary, predictions=preds_binary))

    
if __name__ == "__main__":
    main()

