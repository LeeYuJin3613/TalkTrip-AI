import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# ──────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────
with open(r"C:\Users\dldbw\Desktop\classification_data.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 파일별 그룹화
file_groups = defaultdict(list)
for item in raw_data:
    file_groups[item["source"]].append(item)

# 슬라이딩 윈도우
def make_context_data(file_groups, window=2):
    contexts = []
    labels   = []
    for source, items in file_groups.items():
        for i, item in enumerate(items):
            start      = max(0, i - window)
            prev_texts = [items[j]["text"] for j in range(start, i)]
            curr_text  = item["text"]
            if prev_texts:
                context = " [SEP] ".join(prev_texts) + " [SEP] " + curr_text
            else:
                context = curr_text
            contexts.append(context)
            labels.append(item["label"])
    return contexts, labels

contexts, labels = make_context_data(file_groups, window=2)
print(f"전체: {len(contexts)}개")
print(f"여행(1): {sum(labels)}개")
print(f"일반(0): {len(labels) - sum(labels)}개")

# ──────────────────────────────────────────
# 2. 데이터 분리
# ──────────────────────────────────────────
train_ctx, temp_ctx, train_labels, temp_labels = train_test_split(
    contexts, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels
)
val_ctx, test_ctx, val_labels, test_labels = train_test_split(
    temp_ctx, temp_labels,
    test_size=0.5,
    random_state=42,
    stratify=temp_labels
)

print(f"\n학습:   {len(train_ctx)}개")
print(f"검증:   {len(val_ctx)}개")
print(f"테스트: {len(test_ctx)}개")

# ──────────────────────────────────────────
# 3. 토크나이저 & 데이터셋
# ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

class ContextDataset(Dataset):
    def __init__(self, contexts, labels):
        self.encodings = tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
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

train_dataset = ContextDataset(train_ctx,   train_labels)
val_dataset   = ContextDataset(val_ctx,     val_labels)
test_dataset  = ContextDataset(test_ctx,    test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# ──────────────────────────────────────────
# 4. 모델
# ──────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    "klue/roberta-base",
    num_labels=2
)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
model.to(device)
print(f"\n사용 디바이스: {device}")

# ──────────────────────────────────────────
# 5. 얼리스탑 설정
# ──────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience    # 몇 에폭 기다릴지
        self.min_delta  = min_delta   # 최소 개선량
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc

        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  얼리스탑 카운터: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

        else:
            self.best_score = val_acc
            self.counter    = 0

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# ──────────────────────────────────────────
# 6. 학습 (최대 100 에폭)
# ──────────────────────────────────────────
train_losses = []
val_losses   = []
val_accs     = []
best_val_acc = 0

for epoch in range(100):  # 최대 100 에폭
    # 학습
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 검증
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()
            preds     = torch.argmax(outputs.logits, dim=1)
            correct  += (preds == labels).sum().item()
            total    += len(labels)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = correct / total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    # 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("./model1_best")
        tokenizer.save_pretrained("./model1_best")
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} ← best")
    else:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    # 얼리스탑 체크
    early_stopping(val_acc)
    if early_stopping.stop:
        print(f"\n얼리스탑! {epoch+1} 에폭에서 중단")
        print(f"최고 검증 정확도: {best_val_acc:.4f}")
        break

# ──────────────────────────────────────────
# 7. 테스트
# ──────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained("./model1_best")
model.to(device)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n=== 테스트 결과 ===")
print(classification_report(
    all_labels, all_preds,
    target_names=["일반(0)", "여행(1)"]
))

# ──────────────────────────────────────────
# 8. 시각화
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
axes[0].plot(val_losses,   label='Val Loss',   marker='o', markersize=3)
axes[0].set_title('Loss 변화')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(val_accs, label='Val Accuracy', marker='o', markersize=3, color='green')
axes[1].axhline(y=best_val_acc, color='red', linestyle='--', label=f'최고: {best_val_acc:.4f}')
axes[1].set_title('검증 정확도 변화')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
axes[1].legend()
axes[1].grid(True)

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["일반(0)", "여행(1)"],
    yticklabels=["일반(0)", "여행(1)"],
    ax=axes[2]
)
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('예측')
axes[2].set_ylabel('실제')

plt.tight_layout()
plt.savefig('model1_results.png', dpi=150)
plt.show()
print("\n시각화 저장 완료: model1_results.png")

# ──────────────────────────────────────────
# 9. 실제 문장 테스트
# ──────────────────────────────────────────
def predict(text, prev_texts=None):
    if prev_texts:
        context = " [SEP] ".join(prev_texts) + " [SEP] " + text
    else:
        context = text

    inputs = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred    = torch.argmax(outputs.logits, dim=1).item()
        prob    = torch.softmax(outputs.logits, dim=1)[0]

    label = "여행" if pred == 1 else "일반"
    print(f"문장: {text}")
    print(f"예측: {label} (여행 확률: {prob[1]:.2%})")
    print()

print("\n=== 실제 문장 테스트 ===")

# 문맥 없이
predict("오 좋다 언제 갈까")

# 문맥 있이
predict("오 좋다 언제 갈까", prev_texts=["제주도 가자!", "맞아 좋지"])
predict("좋아!", prev_texts=["치킨 먹을까?", "배고파"])
predict("좋아!", prev_texts=["제주도 가자!", "거기 어때?"])