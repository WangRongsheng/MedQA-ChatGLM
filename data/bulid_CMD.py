import csv
import json
from tqdm import tqdm

data = []
with open('waike.txt', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in tqdm(reader):
        #print(row['title'], row['ask'], row['answer'])
        info = {
           "instruction": str(row['title']),
           "input": str(row['ask']),
           "output": str(row['answer'])
            }
        data.append(info)

with open('waike.json', 'w+', encoding='utf-8') as f:
    json.dump(data, f)