import json
from sklearn.model_selection import train_test_split

data_path = "bert_ner.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
print(f"📊 총 학습 문장 수 : {total:,} 개\n")

# train_ner.py와 동일한 분할 기준으로 계산
train_val, test = train_test_split(data, test_size=0.15, random_state=42)
relative_val = 0.15 / (1 - 0.15)
train, val = train_test_split(train_val, test_size=relative_val, random_state=42)

print(f"Train set     : {len(train):,} 개  ({len(train)/total*100:.1f}%)")
print(f"Validation set: {len(val):,} 개   ({len(val)/total*100:.1f}%)")
print(f"Test set      : {len(test):,} 개   ({len(test)/total*100:.1f}%)")
print(f"{'─' * 40}")
print(f"합계          : {len(train) + len(val) + len(test):,} 개")