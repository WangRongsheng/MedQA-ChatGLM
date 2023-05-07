import json
from tqdm import tqdm

# 打开json文件
with open('train_data.json', 'r', encoding='UTF-8') as f:
    # 从文件读取JSON数据
    data = json.load(f)
# 输出读取的数据
#print(data[0])
#print(data[0][0])
#print(len(data))

data_pre = []
for i in tqdm(range(len(data))):
    ins = str(data[i][0]).replace('病人：', '')
    out = str(data[i][0]).replace('医生：', '')
    info = {
           "instruction": str(ins),
           "input": "",
           "output": str(out)
            }
    data_pre.append(info)

with open('train.json', 'w+', encoding='utf-8') as f:
    json.dump(data_pre, f)