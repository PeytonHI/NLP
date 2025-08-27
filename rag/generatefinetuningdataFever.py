import json
import ollama
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, util

# Initialize a similarity model
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_similar(original_claim, new_claim, threshold=0.8): # generates about 25% of existing data
    similarity = util.pytorch_cos_sim(
        model.encode(original_claim, convert_to_tensor=True),
        model.encode(new_claim, convert_to_tensor=True)
    )
    return similarity.item() > threshold


def generate_claim(claim):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": f"Given the following claim, generate a similar claim. \nClaim: {claim}\nOnly reply with the new claim and nothing else."
            }
        ]
    )
    return response

def generate_label(old_claim, new_claim, old_label):
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": f"Given the following old claim and old label, determine which category the new claim likely belongs in: SUPPORTS, REFUTES, or NOT ENOUGH INFO. \nold_claim: {old_claim}\nold_label:{old_label}\nnew_claim: {new_claim}\n Only reply with the category and nothing else."
            }
        ]
    )
    return response

# resp = generate_claim('dogs are dogs')["message"]["content"]
# print(resp)

with (open(r"C:\Users\peyto\Desktop\school24\497\hw3\data\fever_train.jsonl", encoding='UTF-8')) as f:
  lines = f.readlines()
dataList = [json.loads(line) for line in lines]

# datamini = dataList[:10]
cv_pair = []
print("Working.. but tqdm is broken. Please wait.")

for data in tqdm(dataList):
    old_label = data['label']
    old_claim = data['claim']
    resp = generate_claim(old_claim)
    new_claim = resp["message"]["content"]
    similar_claim = is_similar(old_claim, new_claim)
    if similar_claim == True:
        resp = generate_label(old_claim, new_claim, old_label)
        new_label = resp["message"]["content"]
        cv_pair.append((new_claim, new_label))

with open("cv_pairs.tsv", "w") as fout:
  for c, v in cv_pair:
      j = [{'role': 'user', 'content': c}, {'role': 'assistant', 'content': v} ]
      print(json.dumps(j), sep='\t', file=fout)

print("Document created")
