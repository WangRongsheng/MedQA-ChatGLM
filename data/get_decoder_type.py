import chardet

with open('answers.csv', 'rb') as f:
    result = chardet.detect(f.read())
print(result['encoding'])