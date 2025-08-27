import json
import ollama
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
import time
import requests
import spacy


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
    search_results = response.json()
    # print(search_results)

    # Extract the title of the first result
    # if 'query' in search_results and 'search' in search_results['query']:
    #     title = search_results['query']['search'][0]['title']
    #     return title
    # else:
    #     return None

    # print("Search Results:")
    titles = []
    for result in search_results.get("query", {}).get("search", []):
        # print(f"- {result['title']}")
        titles.append(result)
    
    return titles


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

    # return page_data

def extract_html_info(html_content):
    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract paragraphs
    paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
    
    # print(paragraphs)
    # Extract infobox (if present)
    # infobox = soup.find("table", {"class": "infobox"})
    # infobox_data = {}
    # if infobox:
    #     for row in infobox.find_all("tr"):
    #         header = row.find("th")
    #         value = row.find("td")
    #         if header and value:
    #             infobox_data[header.get_text().strip()] = value.get_text().strip()
    
    # Extract headings and their content
    # headings = []
    # for heading in soup.find_all(["h2", "h3", "h4"]):
    #     heading_text = heading.get_text().strip()
    #     headings.append(heading_text)
    
    return paragraphs


def generate_label(claim, evidence):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": f"Given the following documents: {evidence}, determine if the following claim is true: {claim} only reply with Supports, Refutes, or Not Enough Info and nothing else"
            }
        ]
    )
    return response


def is_similar_query(model, claim, content):
    content_tensor = model.encode(content, convert_to_tensor=True)
    claim_tensor = model.encode(claim, convert_to_tensor=True)
    # aggregated_claim = claim_tensor.mean(dim=0, keepdim=True)
    aggregated_content = content_tensor.mean(dim=0, keepdim=True)
    print(claim_tensor.shape)
    print(aggregated_content.shape)
    similarity = util.pytorch_cos_sim(claim_tensor, aggregated_content)
    similarity.shape

    return similarity.item()

def getTopArticles(articles, k=3):
    return sorted(articles, key=articles.get, reverse=True)[:k]


def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Define the claim
    # claim = "The ice caps are melting" 

    with (open(r"C:\Users\peyto\Desktop\school24\497\final\data\fever_train.jsonl", encoding='UTF-8')) as f:
        lines = f.readlines()
    dataList = [json.loads(line) for line in lines]
    datamini = dataList[:10]


    cv_pair = []
    print("Working.. but tqdm is broken. Please wait.")
    for data in tqdm(datamini):
        # old_label = data['label']
        claim = data['claim']

        # Process the sentence
        doc = nlp(claim)

        entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "WORK_OF_ART")]
        # query = " ".join(entities)  

        # Extract keywords (nouns, proper nouns, etc.)
        # keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
        # query = " ".join(keywords)  

        # print("Extracted Keywords:", keywords)
        # query = " ".join(keyword for keyword in keywords)
        # print("Query:", query)
        # query = "The ice caps are melting"

        entity_to_top_docs = {}
        for entity in entities:
            similar_index = {}
            wiki_search_titles = search_wikipedia(entity)
            
            for data_point in wiki_search_titles:
                title = data_point.get('title')
                html_evidence = get_wikipedia_page(title)
                evidence_doc = extract_html_info(html_evidence)
                
                # Calculate similarity
                similarity = is_similar_query(model, entity, evidence_doc)
                similar_index[title] = similarity

            # Keep only top k relevant documents for this entity
            top_titles = getTopArticles(similar_index, k=3)
            entity_to_top_docs[entity] = {title: similar_index[title] for title in top_titles}
        
        # Compare entity relevance to other entities' top documents
        big_list = []
        for entity, top_docs in entity_to_top_docs.items():
            for other_entity, other_top_docs in entity_to_top_docs.items():
                if entity != other_entity:  # Skip self-comparison
                    for title, similarity in other_top_docs.items():
                        # Compare the current entity's documents to the other entity's title
                        cross_similarity = is_similar_query(model, entity, title)
                        big_list.append((entity, other_entity, title, cross_similarity))
        print(big_list)
        
        sorted_top_k = sorted(big_list, key=lambda x: x[1], reverse=True)[:3]
        print(sorted_top_k)

    

        evidence_doc_list = []
        for title, score in sorted_top_k:
            evidence_docs_dict = {}
            # print('title: ', title)
            html_evidence = get_wikipedia_page(title)
            evidence_doc = extract_html_info(html_evidence)
            # print(evidence_doc)
            evidence_docs_dict[title] = evidence_doc
            evidence_doc_list.append(evidence_docs_dict)
            
        # print(evidence_docs)
        # print(potential_evidence)
        # response = generate_label(claim, evidence_doc_list)
        # label = response["message"]["content"]
        # print(claim, label)

        response = generate_label(claim, evidence_doc_list)
        label = response["message"]["content"]
        cv_pair.append((claim, label))

    with open("cv_pairs_custom.tsv", "w") as fout:
      for c, v in cv_pair:
          j = [{'role': 'user', 'content': c}, {'role': 'assistant', 'content': v} ]
          print(json.dumps(j), sep='\t', file=fout)

    print("Document created")

if __name__ == '__main__':
    main()