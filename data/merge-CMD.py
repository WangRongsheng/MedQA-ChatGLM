import json
import os

# 获取当前文件夹下所有.json文件路径
file_list = []
for file in os.listdir():
    if file.endswith('.json'):
        file_list.append(file)

# 新建合并后的字典
merged_data = []
# 逐个读取json文件并合并到merged_data中
for file in file_list:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        merged_data.append(data)

# 写入新的json文件
with open('merged-CMD.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
print('合并完成！')