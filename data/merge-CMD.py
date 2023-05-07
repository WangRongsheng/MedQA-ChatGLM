import json

# 读取第一个JSON文件
with open('merged4.json', 'r') as f1:
    lst1 = json.load(f1)
# 读取第二个JSON文件
with open('zhongliu.json', 'r') as f2:
    dict2 = json.load(f2)
# 将字典dict2合并到lst1中
lst1.append(dict2)
# 将合并后的数据写入新的JSON文件
with open('merged-CMD.json', 'w') as f:
    json.dump(lst1, f, indent=4)