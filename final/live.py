import json
import ollama
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import requests
import spacy
from sklearn.metrics import classification_report
import regex as re

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
            
# create labels based on given claims and documents using ollama
def generate_label(claim, documents):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": f"Based on the following documents, determine if the following claim is true. \nDocuments: {documents}\nClaim: {claim}\nOnly reply with SUPPORTS, REFUTES, or NOT ENOUGH INFO and nothing else. Choose the label with the strongest evidence"
            }
        ]
    )
    return response

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

    aggregated_content = content_tensor.mean(dim=0, keepdim=True) # convert ([[14, 384]]) -> ([[1, 384]]) so we dont get a tensor for each paragraph
    aggregated_content.to(device)
    # print('aggregated_content shape :', aggregated_content.shape)

    if claim_tensor.shape != aggregated_content.shape:
        return None

    similarity = util.pytorch_cos_sim(claim_tensor, aggregated_content)
    # print('sim shape :', similarity.shape)

    return similarity.item()


def getTopArticles(articles, k=1):
    return sorted(articles, key=articles.get, reverse=True)[:k]


def extract_first_verb_phrase(doc):
    for token in doc:
        # Check if the token is a verb and if it is the head of a verb phrase
        if token.pos_ == "VERB":
            # Collect the verb phrase (verb + its children)
            verb_phrase = " ".join([child.text for child in token.children if child.dep_ in ("dobj", "nsubj", "prep") or child == token])
            return verb_phrase.strip() if verb_phrase else token.text  # Return the verb or the constructed phrase

    return None  # Return None if no verb found


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to('cpu')
    nlp = spacy.load("en_core_web_sm")

    while True:
        claim = input('Enter claim...')
    # load spacy model
    
    # with (open(r"C:\Users\peyto\Desktop\school24\497\final\data\fever_train.jsonl", encoding='UTF-8')) as f:
    #     lines = f.readlines()
    # dataList = [json.loads(line) for line in lines]
    # datamini = dataList[:1000]

    # count = 0
    # cv_pair = []
    # refs_binary = []
    # preds_binary = []
    # claim = data['claim']
    # true_label = data['label']
    # label_map = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
    # refs_binary.append(label_map[true_label])

        # pass claim to spacy
        doc = nlp(claim)

        # extract keywords (nouns, proper nouns, etc.)
        # keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        # claim = " ".join(keywords)  q

        # entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART")]
        # query = " ".join(entities)  

        # proc_claim = extract_first_verb_phrase(doc)
        # if proc_claim == None:
        #     keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        #     proc_claim = " ".join(keywords)


        # print('query: ', claim)

        # grab first 3 wiki results and check their similarity with claim then add to dict
        wiki_search_titles = search_wikipedia(claim)
        similar_index = {}
        for data_point in wiki_search_titles[:3]:
            title = data_point.get('title')
            if title is None:
                break
            print("Wiki title: ", title)
            html_evidence = get_wikipedia_page(title) # retrieve html page
            evidence_doc = extract_html_info(html_evidence) # extract paragraphs (docs)
            if evidence_doc is None:
                break
            relevant_doc = is_similar_query(model, claim, evidence_doc) # check similarity between claim and docs
            if relevant_doc is None:
                break
            similar_index[title] = relevant_doc

        top_k_titles = getTopArticles(similar_index) # select top 3 articles with highest cosine sim scores

        # get wiki results again but from the top 3 pages only
        evidence_doc_list = []
        for title in top_k_titles:
            evidence_docs_dict = {}
            html_evidence = get_wikipedia_page(title)
            evidence_doc = extract_html_info(html_evidence)
            evidence_docs_dict[title] = evidence_doc
            evidence_doc_list.append(evidence_docs_dict)

        response = generate_label(claim, evidence_doc_list) 
        pred_label = response["message"]["content"]
        pred_label = pred_label.strip()
        print('Claim: ', claim, 'Label: ', pred_label)

        # if re.match(r'^\s*[Nn]', pred_label):
        #     pred_label = 'NOT ENOUGH INFO'
        # elif re.match(r'^\s*[Ss]', pred_label):
        #     pred_label = 'SUPPORTS'
        # elif re.match(r'^\s*[Rr]', pred_label):
        #     pred_label = 'REFUTES'

        # preds_binary.append(label_map[pred_label])
        # cv_pair.append((claim, pred_label))
        # data_length = len(datamini)
        # count += 1
        # print(f"Claim added: {count}/{data_length}")

        # with open("custom-multiQe.jsonl", "w") as fout:
        #   for id, (claim, pred_label) in enumerate(cv_pair):
        #       j = {'id': id,'label': pred_label, 'claim': claim}
        #       print(json.dumps(j), file=fout)

        # print("Document created")

        # print(classification_report(refs_binary, preds_binary, target_names=label_map.keys()))

if __name__ == '__main__':
    main()