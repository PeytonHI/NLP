import torch
import json
import random
import evaluate
from collections import Counter
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class MyClassifier(torch.nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, label_count):
        super().__init__()
        # raise NotImplementedError()
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
    
    def to_id(self, tok):
        return self.tok2idx.get(tok, 0)

    def to_tok(self, id):
        return self.idx2tok.get(id, "[UNK]")
    

class LabelVocab:
    def __init__(self, labels):
        self.label2id = {label: idy for idy, label in enumerate(set(labels))}
        self.idy2label = {idy: label for label, idy in self.label2id.items()}

    def to_id (self, label):
        return self.label2id.get(label, 0)
    

class MemeDataSet(Dataset):

    def __init__(self, tokenized_data, vocab, label_vocab, label_count):
        # raise NotImplementedError()
        self.tensor_data = []
        # labels = [label for y, x in tokenized_data for label in y]
        # text = [text for y, x in tokenized_data for text in x]
      
        for y, x in tokenized_data:
            if y:  # Check if y is not empty
                x = torch.LongTensor([vocab.to_id(tok) for tok in x])
                y = torch.tensor(label_vocab.to_id(y[0]))  # Single label per sequence
                # x = F.pad(x, (0, 100 - x.size(0)), value=vocab.to_id("[PAD]"))

                # x_padded = pad_sequence(x, batch_first=True, padding_value=0)
                # y_padded = pad_sequence(y, batch_first=True, padding_value=0)

                # self.tensor_data.append((x_padded, y_padded))
            else:
                print(f"Skipping entry with empty label: {x}")
            
            self.tensor_data.append((x, y))
        
        
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        return self.tensor_data[idx]
    

def pad_collate(batch):
    (xx, yy) = zip(*batch)
    # xx = [x for x in xx if x.numel() > 0]
    # yy = [y for i, y in enumerate(yy) if xx[i].numel() > 0]  # Align labels with non-empty sequences

    # if not xx:
    #     return torch.empty(0), torch.empty(0) 
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_tensor = torch.tensor(yy)  # Convert labels to tensor

    return xx_pad, yy_tensor


def load_data():
    # raise NotImplementedError()

    # tokenizer = Tokenizer.from_pretrained("bert-base-cased")

    with open(r"\Users\peyto\Desktop\school24\497\hw2-neural-models-peyton-master\scorer\data\train.json", 'r', encoding="UTF-8") as f:
        train = []
        data = json.load(f)
      
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
    loss_func = F.cross_entropy  # same as torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1):
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

        # after each epoch, check how we're doing
        # compute avg loss over train and dev sets
        with torch.no_grad():
            total_loss = 0
            for x, y in tqdm(train_data):
                pred = model(x)
                y = y.view(-1)
                loss = loss_func(pred, y)
                total_loss += loss
            print("train loss:", total_loss / len(train_data))

            total_loss = 0
            for x, y in tqdm(val_data):
                pred = model(x)
                loss = loss_func(pred, y)
                total_loss += loss
            print("dev loss:", total_loss / len(val_data))


def run_model_on_dev_data(val_data, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for x, y in val_data:
            # print(x.shape)
            pred = model(x)  # pred is something like [0.6, 0.4]
            # print(pred)
            # TODO: when using batched inputs, your output will also be batched
            # so you need to split them before appending to preds
            # softmax_pred = F.softmax(pred, dim=1)
            # pred_labels = torch.argmax(pred, dim=1)  
            probabilities = torch.sigmoid(pred)
            preds.append(probabilities)
        

    return preds


def sample_predictions(preds, val_data_raw):
    for _ in range(5):
        idx = random.randint(0, len(val_data_raw) -1)
        pred_labels = None
        # pred_label_id = torch.argmax(preds[idx]).item()
        probabilities = torch.sigmoid(preds[idx])

        threshold = 0.5
        pred_labels = (probabilities > threshold).int() # This is the list of labels
    
        gold_labels = val_data_raw[idx][0]

        print("Input:", " ".join(val_data_raw[idx][1]))
        print("Gold: ", gold_labels)

        # preds are not normalized, so for better viewing, run it through softmax
        print("Pred: ", pred_labels, probabilities) 
        print() 

    return gold_labels, pred_labels       

# def pad_func(dataloader, voc_length):
    # embedding = torch.nn.Embedding(voc_length, 50)
    # for (x_padded, y_padded, x_lens, y_lens) in      enumerate(dataloader):
    #     x_embed = embedding(x_padded)
    #     x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)
    
    # # rnn = torch.nn.GRU(100, 100, 3, batch_first=True)
    # x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False) 
    # output_packed, hidden = MyClassifier(voc_length,x_packed, hidden)

    # output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)

    # return output_padded, output_lengths




def main():

    train_data_raw, val_data_raw = load_data()

    vocab = Vocab([word
                for y, x in train_data_raw
                for word in x])
    label_vocab = LabelVocab([word for y, x in train_data_raw for word in y])

    train_label_list = [label for y, x in train_data_raw for label in y]
    val_label_list = [label for y, x in train_data_raw for label in y]

    train_label_count = len(set(train_label_list))
    val_label_count = len(set(val_label_list))

    # print(type(label_vocab), label_vocab)
    train_dataset = MemeDataSet(train_data_raw, vocab, label_vocab, train_label_count)
    val_dataset = MemeDataSet(val_data_raw, vocab, label_vocab, val_label_count)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=pad_collate)

    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    # for batch in dataloader:
    #     texts, labels = batch
    #     print(f'Texts: {texts}')
    #     print(f'Labels: {labels}')
    voc_length = len(vocab)

    # train_output_padded, train_output_lengths = pad_func(train_dataloader, voc_length)

    # val_output_padded, val_output_lengths = pad_func(val_dataloader, voc_length)
   
    model = MyClassifier(voc_length, 100, 100, train_label_count)
    
    train(model, train_dataloader, val_dataloader)

    preds = run_model_on_dev_data(val_dataset, model)
    gold_labels, pred_labels = sample_predictions(preds, val_data_raw)

    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    accuracy = evaluate.load("accuracy")

    refs = []
    preds_binary = []

    for i, (labels, _) in enumerate(val_data_raw):
        refs.append([label_vocab.to_id(label) for label in labels])  

        preds_binary.append([label_vocab.to_id(label) for label in (preds[i] > 0.5).int().tolist()])


    print(precision.compute(references=refs, predictions=preds_binary))
    print(recall.compute(references=refs, predictions=preds_binary))
    print(accuracy.compute(references=refs, predictions=preds_binary))

    # TODO: evaluate your model on the validation data and print metrics

    # you can structure these functions however you wish
    # just make sure to print out precision and recall at the end

    
if __name__ == "__main__":
    main()