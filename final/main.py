import json
import spacy
from tqdm.notebook import tqdm
from encoder import BiLSTMEncoder
from classifier import Classifier
from utils import is_similar_query
import json
import ollama
import torch
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import time
import requests
import spacy
from tokenizers import Tokenize

from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# Sample vocabulary for tokenization
vocab = defaultdict(lambda: len(vocab))  # Creates unique indices for words
UNK = vocab["<UNK>"]  # Unknown token

# Tokenizer
def tokenize(text):
    tokens = text.lower().split()  # Simple whitespace tokenizer
    return [vocab[token] for token in tokens]


def prepare_input(texts, max_length=50):
    tokenized = [torch.tensor(tokenize(text), dtype=torch.long) for text in texts]  # Use torch.long
    padded = pad_sequence(tokenized, batch_first=True, padding_value=0)
    return padded[:, :max_length]



# query and return titles associated with query
def search_wikipedia(query):
    # Search Wikipedia
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": 5

    }

    response = requests.get(search_url, params=params)
    assert response.status_code == 200, f'Error querying wikipedia {response.status_code}'
    print("Successful wiki query")
    search_results = response.json()
    
    titles = []
    for result in search_results.get("query", {}).get("search", []):
        titles.append(result)
    
    return titles

# get page data by passing in each title and return html data for page
def get_wikipedia_page(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": 2,
        "utf8": 1
    }

    response = requests.get(url, params=params)
    page_data = response.json()

    # Extract the page summary
    if 'parse' in page_data:
        return page_data['parse']['text']
    else:
        return None

# parse html with BeautifulSoup
def extract_html_info(html_content):
    if html_content is None:
      return None
    soup = BeautifulSoup(html_content, "html.parser")
    
    # get paragraphs
    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    
    return paragraphs
        

# sim index between claim and all content (articles), 
def is_similar_query(model, claim, content):
    device = torch.device("cpu")

    claim_tensor = model.encode(claim, convert_to_tensor=True)
    claim_tensor = claim_tensor.unsqueeze(0)
    claim_tensor.to(device)
    # print('claim_tensor shape :', claim_tensor.shape)

    content_tensor = model.encode(content, convert_to_tensor=True)
    content_tensor.to(device)    
    # print('content shape :', content_tensor.shape)

    aggregated_content = content_tensor.mean(dim=0, keepdim=True) # convert ([[14, 384]]) -> ([[1, 384]])
    aggregated_content.to(device)
    print('aggregated_content shape :', content_tensor.shape)

    # if claim_tensor.shape != aggregated_content.shape:
    #     return None

    similarity = util.pytorch_cos_sim(claim_tensor, aggregated_content)
    # print('sim shape :', similarity.shape)

    return similarity.max().item()


def getTopArticles(articles, k=3):
    return sorted(articles, key=articles.get, reverse=True)[:k]


def main():
    # Example vocabulary size (this should match your actual vocabulary)
    vocab = tokenize.get_vocab()
    vocab_size = len(vocab)  # This should be used to initialize your encoder


    embedding_dim = 300
    hidden_dim = 512
    bilstm_encoder = BiLSTMEncoder(vocab_size, embedding_dim, hidden_dim).to('cpu')


    # Load dataset
    with open(r"C:\Users\peyto\Desktop\school24\497\final\data\fever_train.jsonl", encoding='UTF-8') as f:
        lines = f.readlines()
    dataList = [json.loads(line) for line in lines]
    datamini = dataList[:10]

    # Initialize NLP model
    nlp = spacy.load("en_core_web_sm")

    # Prediction process
    predictions = []
    label_map = {0: 'NOT ENOUGH INFO', 1: 'SUPPORTS', 2: 'REFUTES'}
    for data in tqdm(datamini):
        claim = data['claim']

        # Tokenize and prepare input for the encoder
        claim_tensor = prepare_input([claim])  # Type torch.long
        claim_encoding = bilstm_encoder(claim_tensor)  # Output will be torch.float32


        # Retrieve evidence
        doc = nlp(claim)
        keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        query = " ".join(keywords)

        wiki_search_titles = search_wikipedia(query)
        similar_index = {}
        for data_point in wiki_search_titles[:1]:
            title = data_point.get('title')
            if not title:
                continue

            html_evidence = get_wikipedia_page(title)
            evidence_doc = extract_html_info(html_evidence)
            if not evidence_doc:
                continue

            evidence_tensor = prepare_input(evidence_doc)
            evidence_encoding = bilstm_encoder(evidence_tensor)

            relevant_score = is_similar_query(bilstm_encoder, claim, evidence_doc)
            similar_index[title] = relevant_score

        # Select top articles
        top_k_titles = getTopArticles(similar_index)
        evidence_doc_list = [extract_html_info(get_wikipedia_page(title)) for title in top_k_titles]

        # Classify claim with evidence
        evidence_tensors = prepare_input([" ".join(doc) for doc in evidence_doc_list])
        evidence_encodings = bilstm_encoder(evidence_tensors)

        outputs = Classifier(claim_encoding, evidence_encodings)
        predicted_label = torch.argmax(outputs, dim=-1).item()

        # Map label to string
        predictions.append({
            'claim': claim,
            'predicted_label': label_map[predicted_label]
        })

    # Save predictions
    with open("predictions.jsonl", "w") as fout:
        for prediction in predictions:
            json.dump(prediction, fout)
            fout.write('\n')

    print("Predictions saved in 'predictions.jsonl'")

if __name__ == '__main__':
    main()
