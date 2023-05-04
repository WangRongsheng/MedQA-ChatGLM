# 关于编码问题解决：https://blog.csdn.net/AlAuAu/article/details/109478113
import pandas as pd
import os
import json
from tqdm import tqdm

# 指定文件夹路径
folder_path = "./"
# 读取question.csv文件
questions_df = pd.read_csv(os.path.join(folder_path, "questions.txt"))
# 读取answer.csv文件
answers_df = pd.read_csv(os.path.join(folder_path, "answers.txt"))

merged = pd.merge(questions_df, answers_df, on='que_id')
#print(len(merged))
#print(merged.columns)

# 存储数组
data = []

# 遍历merged对象的每一行数据
for index, row in tqdm(merged.iterrows()):
    # 获取每一列的数据
    q_id = row['que_id']
    q_content = row['content_x']
    
    a_id = row['ans_id']
    a_content = row['content_y']
    
    #print(q_id)
    #print(q_content)
    #print(a_id)
    #print(a_content)
    #break
    
    # 保存json
    info = {
           "instruction": str(q_content),
           "input": "",
           "output": str(a_content)
            }
    data.append(info)

with open('cMedQA.json', 'w+', encoding='utf-8') as f:
    json.dump(data, f)