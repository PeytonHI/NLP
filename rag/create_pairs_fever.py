import json
from tqdm.notebook import tqdm

with (open(r"C:\Users\peyto\Desktop\school24\497\hw3\data\fever_train.jsonl", encoding='UTF-8')) as f:
  lines = f.readlines()
dataList = [json.loads(line) for line in lines]

cv_pair = []
print("Working.. but tqdm is broken. Please wait.")
for data in tqdm(dataList):
    claim = data['claim']
    label = data['label']
    cv_pair.append((claim, label))

with open("cv_pairs_fever.tsv", "w") as fout:
  for c, v in cv_pair:
      j = [{'role': 'user', 'content': c}, {'role': 'assistant', 'content': v} ]
      print(json.dumps(j), sep='\t', file=fout)

print("Document created")
