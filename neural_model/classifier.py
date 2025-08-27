import torch
import json
import random
import evaluate
from collections import Counter
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

# Author: Peyton

class MyClassifier(torch.nn.Module):   

    def __init__(self, voc_size, emb_size, hidden_size, label_count):
        super().__init__()
        self.emb = torch.nn.Embedding(voc_size, emb_size)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(hidden_size * 2, label_count)


    def forward(self, input):
        e = self.emb(input)
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


class Vocab:
    def __init__(self, tokens):
        self.vocab = [tok for tok, count in Counter(tokens).most_common()]
        self.tok2idx = {tok: idx + 2 for idx, tok in enumerate(self.vocab)}
        self.tok2idx[0] = "[PAD]"
        self.tok2idx[1] = "[UNK]"
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}
    
    def __len__(self):
        return len(self.tok2idx)
    
    def to_idx(self, tok):
        return self.tok2idx.get(tok, 0)

    def to_tokx(self, idx):
        return self.idx2tok.get(idx, "[UNK]")
    

class LabelVocab:
    def __init__(self, labels):
        self.label_vocab = set(labels)
        self.label2idy = {label: idy + 2 for idy, label in enumerate(self.label_vocab)}
        self.label2idy[0] = "[PAD]"
        self.label2idy[1] = "[UNK]"
        self.idy2label = {idy: label for label, idy in self.label2idy.items()}

    def __len__(self):
            return len(self.label2idy)
    
    def to_idy(self, label):
        return self.label2idy.get(label, 0)
    
    def to_toky(self, idy):
        return self.idy2label.get(idy, "[UNK]")

class MemeDataSet(Dataset):

    def __init__(self, tokenized_data, vocab, label_vocab, label_count):
        # raise NotImplementedError()
        self.tensor_data = []
      
        for y, x in tokenized_data:
            if y:  # Check if y is not empty
                x_tensor = torch.LongTensor([vocab.to_idx(tok) for tok in x])
                y_tensor = torch.zeros(label_count)  # Use float for multi-label
                for label in y:
                    if label in label_vocab.label2idy:
                        y_tensor[label_vocab.label2idy[label]] = 1.0

                self.tensor_data.append((x_tensor, y_tensor))
            else:
                print(f"Skipping entry with empty label: {x_tensor}")
            
        
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        return self.tensor_data[idx]
    

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_tensor = torch.stack(yy)

    return xx_pad, yy_tensor


def load_data():

    with open(r"\Users\peyto\Desktop\school24\497\hw2-neural-models-peyton-master\scorer\data\train.json", 'r', encoding="UTF-8") as f:
        train = []
        
        data = json.load(f)
        # tokenizer = Tokenizer.from_pretrained("bert-base-cased")

        for item in data:
            text = item['text']
            label = item['labels']
            # tokens = tokenizer.encode(text).tokens
            tokens = text.split()
            if label and tokens:  # Ensure both label and tokens are non-empty
                train.append((label, tokens))
            else:
                print(f"Skipping entry with empty label or tokens: {text}")


    with open(r"\Users\peyto\Desktop\school24\497\hw2-neural-models-peyton-master\scorer\data\validation.json", 'r', encoding="UTF-8") as f2:
        val = []
        data = json.load(f2)

        for item in data:
            text = item['text']
            label = item['labels']
            # tokens = tokenizer.encode(text).tokens
            tokens = text.split()
            if label and tokens:  # Ensure both label and tokens are non-empty
                val.append((label, tokens))            
            else:
                print(f"Skipping entry with empty label or tokens: {text}")
            
    return train, val

    
def train(model, train_data, val_data):
    # raise NotImplementedError()
    # setup the training
    loss_func = torch.nn.BCEWithLogitsLoss()  # Use BCE with logits for multi-label classification
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(2):
        print("Epoch", epoch)
        
        for x, y in tqdm(train_data):

            model.zero_grad()  # do this before running
            # print("Batch input shape:", x.shape)  # Should be (4, seq_length)
            # print("Batch target shape:", y.shape)  # Should match (4,) or (4, label_count)

            pred = model(x)
            # print("Prediction shape:", pred.shape)  # Should be (4, label_count)
            loss = loss_func(pred, y)
            loss.backward()  # calculate gradients
            optimizer.step()  # updates thetas
            # print(f"Loss: {loss.item()}")


        # after each epoch, check how we're doing
        # compute avg loss over train and dev sets
        with torch.no_grad():
            total_loss = 0
            for x, y in tqdm(train_data):
                pred = model(x)
                loss = loss_func(pred, y)
                total_loss += loss
            print("train loss:", total_loss / len(train_data))

            total_loss = 0
            for x, y in tqdm(val_data):
                pred = model(x)
                loss = loss_func(pred, y)
                # print(f"Loss: {loss.item()}")

                total_loss += loss
            print("dev loss:", total_loss / len(val_data))


def run_model_on_dev_data(val_data, model):
    preds = []
    refs = []
    model.eval()
    with torch.no_grad():
        print(f"Number of batches in val_data: {len(val_data)}")
        for x, y in val_data:
            # print(x.shape)
            print("x shape in val data", x.shape, "y shape:", y.shape)
            pred = model(x)  # pred is something like [0.6, 0.4]
            # print(pred)
            # TODO: when using batched inputs, your output will also be batched
            # so you need to split them before appending to preds

            print(f"Predictions shape: {pred.shape}, References shape: {y.shape}")  # For debugging 
            print(f"Raw prediction: {pred}")

            probabilities = torch.sigmoid(pred)
            print("Probs:", probabilities[:5])  # Debugging: check first few predicted probabilities

            preds.append((probabilities > 0.2).int())     
            refs.append(y)

    preds = torch.cat(preds, dim=0)  # Shape [total_samples, num_labels]
    refs = torch.cat(refs, dim=0)
    print(f"Predictions shape: {preds.shape}, References shape: {refs.shape}")  # For debugging 

    return preds, refs


def sample_predictions(preds, dev_data_raw, label_vocab):
    for _ in range(5):
        pred_labels = []

        idx = random.randint(0, len(dev_data_raw) - 1)
        # text = vocab.to_tok(idx)
        gold_text = dev_data_raw[idx][1]
        # gold_label = label_vocab.idy2label[idx]
        gold_label = dev_data_raw[idx][0]

        for i, pred in enumerate(preds[idx]):
                if pred == 1:
                    label = label_vocab.idy2label[i]
                    pred_labels.append(label)
        
        print("Input: ", gold_text)
        print("Gold label:", gold_label)
        print("Predicted labels: ", pred_labels)


def main():

    train_data_raw, val_data_raw = load_data()
    # print(train_data_raw[:5])
    # print(val_data_raw[:5])

    vocab = Vocab([word
                for y, x in train_data_raw
                for word in x])
    
    label_vocab = LabelVocab([label for y, x in train_data_raw for label in y])

    # train_label_list = [label for y, x in train_data_raw for label in y]
    # val_label_list = [label for y, x in val_data_raw for label in y]

    # train_label_count = len(train_label_list)
    # val_label_count = len(val_label_list)

    unique_train_label_count = len(label_vocab)
    unique_val_label_count = len(label_vocab)

    unique_voc_length = len(vocab)

    train_dataset = MemeDataSet(train_data_raw, vocab, label_vocab,unique_train_label_count)
    val_dataset = MemeDataSet(val_data_raw, vocab, label_vocab, unique_val_label_count)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)
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

    model = MyClassifier(unique_voc_length, 100, 100, unique_train_label_count)
    
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

    sample_predictions(preds, val_data_raw, label_vocab)

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

    # TODO: evaluate your model on the validation data and print metrics

    # you can structure these functions however you wish
    # just make sure to print out precision and recall at the end
    
    # generate and store labels
    new_id_list = []
    json_list = []
    with open("prediction_output.json", "w") as f:
        for _ in range(len(preds)):
            label_list = []
            new_id = 0

            while new_id not in new_id_list:
                new_id = random.randint(0, 100000)
                new_id = str(new_id)
                new_id_list.append(new_id)

            for i, pred in enumerate(preds[_]):
                if pred == 1:
                    label = label_vocab.idy2label[i]
                    label_list.append(label)
            
            json_list.append({
                "id": new_id,
                "labels": label_list
            })

        json.dump(json_list, f, indent=4)
        
    
if __name__ == "__main__":
    main()