"""
eval_testset.py
===============
저장된 모델로 테스트셋 성능을 평가하는 스크립트.

사용법:
  python eval_testset.py

전제조건:
  - 학습 코드(model1v3.py)와 동일한 경로 구조
  - 데이터: C:\\dev\\TalkTrip\\TripTalk_ai_model\\dataset\\test.json
  - 모델: ./model1_v3_dropout04_lr5e6/
"""

import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DATA_PATH  = r"C:\dev\TalkTrip\TripTalk_ai_model\dataset\test.json"
MODEL_DIR  = "./model1_v3_dropout04_lr5e6"
WINDOW     = 3
MAX_LEN    = 384
BATCH_SIZE = 16
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────
# 1. 데이터 로드 & 분리 (학습 코드와 동일한 seed/split)
# ─────────────────────────────────────────
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

file_groups = defaultdict(list)
for item in raw_data:
    file_groups[item["source"]].append(item)

all_sources = list(file_groups.keys())
random.shuffle(all_sources)          # seed=42 고정 → 학습 때와 동일한 순서

n             = len(all_sources)
train_sources = all_sources[:int(n * 0.70)]
val_sources   = all_sources[int(n * 0.70):int(n * 0.85)]
test_sources  = all_sources[int(n * 0.85):]

print(f"전체 파일: {n}개")
print(f"  학습: {len(train_sources)}개 파일")
print(f"  검증: {len(val_sources)}개 파일")
print(f"  테스트: {len(test_sources)}개 파일")
print(f"  테스트 소스: {test_sources}")

# ─────────────────────────────────────────
# 2. 슬라이딩 윈도우
# ─────────────────────────────────────────
def make_split_data(sources, file_groups, window=WINDOW):
    contexts, labels = [], []
    for source in sources:
        items = file_groups[source]
        for i, item in enumerate(items):
            start      = max(0, i - window)
            prev_texts = [items[j]["text"] for j in range(start, i)]
            curr_text  = item["text"]
            end        = min(len(items), i + window + 1)
            next_texts = [items[j]["text"] for j in range(i + 1, end)]
            parts      = prev_texts + [curr_text] + next_texts
            contexts.append(" [SEP] ".join(parts))
            labels.append(item["label"])
    return contexts, labels

test_ctx, test_labels = make_split_data(test_sources, file_groups)
print(f"\n테스트셋: {len(test_ctx)}개 문장 | 여행(1): {sum(test_labels)}개 ({sum(test_labels)/len(test_labels)*100:.1f}%)")

# ─────────────────────────────────────────
# 3. 토크나이저 & 데이터셋
# ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

class ContextDataset(Dataset):
    def __init__(self, contexts, labels):
        self.encodings = tokenizer(
            contexts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

test_dataset = ContextDataset(test_ctx, test_labels)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ─────────────────────────────────────────
# 4. 모델 로드 & 평가
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n사용 디바이스: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

all_preds, all_labels_list = [], []
all_probs = []

loss_fn   = CrossEntropyLoss()
total_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels_batch)
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(labels_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_test_loss = total_loss / len(test_loader)
test_acc      = sum(p == l for p, l in zip(all_preds, all_labels_list)) / len(all_labels_list)

# ─────────────────────────────────────────
# 5. 결과 출력
# ─────────────────────────────────────────
print("\n" + "="*50)
print("=== 테스트셋 평가 결과 ===")
print("="*50)
print(f"Test Loss    : {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print()
print(classification_report(
    all_labels_list, all_preds,
    target_names=["일반(0)", "여행(1)"],
    zero_division=0,
    digits=4
))

# ─────────────────────────────────────────
# 6. 오분류 문장 출력 (상위 20개)
# ─────────────────────────────────────────
print("="*50)
print("=== 오분류 문장 샘플 (최대 20개) ===")
print("="*50)

errors = []
for i, (pred, label, prob, ctx) in enumerate(zip(all_preds, all_labels_list, all_probs, test_ctx)):
    if pred != label:
        # 현재 문장만 추출 (SEP 기준 중간)
        parts   = ctx.split(" [SEP] ")
        mid_idx = min(WINDOW, len(parts) - 1)
        curr    = parts[mid_idx]
        errors.append({
            "idx":    i,
            "text":   curr,
            "actual": label,
            "pred":   pred,
            "prob":   prob
        })

errors.sort(key=lambda x: abs(x["prob"] - 0.5))  # 확신이 강한 오분류 순

for e in errors[:20]:
    actual_str = "여행(1)" if e["actual"] == 1 else "일반(0)"
    pred_str   = "여행(1)" if e["pred"]   == 1 else "일반(0)"
    print(f"[{e['idx']:4d}] 실제:{actual_str} → 예측:{pred_str} (여행확률:{e['prob']:.2%})")
    print(f"       {e['text'][:60]}")
    print()

# ─────────────────────────────────────────
# 7. 시각화 저장
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion Matrix
cm = confusion_matrix(all_labels_list, all_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["일반(0)", "여행(1)"],
    yticklabels=["일반(0)", "여행(1)"],
    ax=axes[0]
)
axes[0].set_title(f'Confusion Matrix\n(Acc: {test_acc:.4f}, Loss: {avg_test_loss:.4f})')
axes[0].set_xlabel('예측')
axes[0].set_ylabel('실제')

# 예측 확률 분포
bins = np.linspace(0, 1, 21)
axes[1].hist(
    [p for p, l in zip(all_probs, all_labels_list) if l == 0],
    bins=bins, alpha=0.6, label='실제 일반(0)', color='steelblue'
)
axes[1].hist(
    [p for p, l in zip(all_probs, all_labels_list) if l == 1],
    bins=bins, alpha=0.6, label='실제 여행(1)', color='tomato'
)
axes[1].axvline(0.5, color='black', linestyle='--', label='결정경계(0.5)')
axes[1].set_title('여행 예측 확률 분포')
axes[1].set_xlabel('여행(1) 확률')
axes[1].set_ylabel('문장 수')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = "./eval_testset_result.png"
plt.savefig(out_path, dpi=150)
plt.show()
print(f"\n시각화 저장 완료: {out_path}")
print(f"오분류 총 {len(errors)}개 / 전체 {len(all_labels_list)}개")
