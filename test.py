import chardet

filename = '你要檢查的檔案路徑'

with open(filename, 'rb') as f:
    rawdata = f.read()

result = chardet.detect(rawdata)
print(f"檔案編碼: {result['encoding']}")
print(f"可信度: {result['confidence']}")
