from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class POSBERT(nn.Module):
    def __init__(self, base_model, num_labels):
        super(POSBERT, self).__init__()
        self.bert = base_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        logits = self.classifier(last_hidden)  
        return logits


class POSDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, label_map, max_length):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag_seq = self.tags[idx]

        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        word_ids = encoding.word_ids(batch_index=0)
        aligned_tags = [
            -100 if wid is None else self.label_map[tag_seq[wid]]
            for wid in word_ids
        ]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tag_ids": torch.tensor(aligned_tags)
        }


def read_conllu(filepath):
    sentences = []
    tags = []
    with open(filepath, "r", encoding="utf-8") as f:
        sentence = []
        tag_seq = []
        for line in f:
            if line.startswith("#") or line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag_seq)
                    sentence = []
                    tag_seq = []
                continue
            parts = line.strip().split("\t")
            if len(parts) == 10:  # Ensure it's a valid token line
                sentence.append(parts[1])  # Word form
                tag_seq.append(parts[3])  # Universal POS tag

    return sentences, tags


def main():
    # Universal POS tags
    num_labels = 17

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", model_max_length=128)
    base_model = AutoModel.from_pretrained("distilbert-base-cased")

    # Read dataset
    filepath = r'C:\Users\peyto\Desktop\school24\497\hw4\en_ewt-ud-train.conllu'
    sentences, tags = read_conllu(filepath)

    # Create label map
    unique_tags = sorted(set(tag for tag_seq in tags for tag in tag_seq))
    label_map = {tag: idx for idx, tag in enumerate(unique_tags)}

    # Create dataset and dataloader
    dataset = POSDataset(sentences, tags, tokenizer, label_map, max_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = POSBERT(base_model, num_labels)

    # Enable gradient checkpointing for memory efficiency
    model.bert.gradient_checkpointing_enable()

    # Process batches
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        tag_ids = batch["tag_ids"]

        # Model inference
        logits = model(input_ids, attention_mask)
        predicted_tags = torch.argmax(logits, dim=-1)


        id_to_label = {idx: tag for tag, idx in label_map.items()}

    decoded_tags = [
        [id_to_label[idx.item()] for idx in sequence]
        for sequence in predicted_tags
    ]

    # Print decoded tags for the first batch
    for sentence_tags in decoded_tags:
        print(sentence_tags)


if __name__ == '__main__':
    main()
