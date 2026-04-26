import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict

# ──────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────
with open(r"C:\dev\TalkTrip\TripTalk_ai_model\dataset\classification_relabeled.json", 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

file_groups = defaultdict(list)
for item in raw_data:
    file_groups[item["source"]].append(item)

# ──────────────────────────────────────────
# 2. 슬라이딩 윈도우 함수
# ──────────────────────────────────────────
def make_split_data(sources, file_groups, window=3):
    contexts = []
    labels   = []

    for source in sources:
        items = file_groups[source]
        for i, item in enumerate(items):
            start      = max(0, i - window)
            prev_texts = [items[j]["text"] for j in range(start, i)]
            curr_text  = item["text"]
            end        = min(len(items), i + window + 1)
            next_texts = [items[j]["text"] for j in range(i + 1, end)]
            parts      = prev_texts + [curr_text] + next_texts
            context    = " [SEP] ".join(parts)
            contexts.append(context)
            labels.append(item["label"])

    return contexts, labels

# ──────────────────────────────────────────
# 3. 파일 단위로 학습·검증·테스트 분리
# ──────────────────────────────────────────
all_sources = list(file_groups.keys())
random.seed(42)
random.shuffle(all_sources)

n             = len(all_sources)
train_sources = all_sources[:int(n * 0.7)]
val_sources   = all_sources[int(n * 0.7):int(n * 0.85)]
test_sources  = all_sources[int(n * 0.85):]

train_ctx, train_labels = make_split_data(train_sources, file_groups)
val_ctx,   val_labels   = make_split_data(val_sources,   file_groups)
test_ctx,  test_labels  = make_split_data(test_sources,  file_groups)

print(f"학습 파일: {len(train_sources)}개 | 문장: {len(train_ctx)}개 | 여행(1): {sum(train_labels)}개")
print(f"검증 파일: {len(val_sources)}개   | 문장: {len(val_ctx)}개   | 여행(1): {sum(val_labels)}개")
print(f"테스트 파일: {len(test_sources)}개 | 문장: {len(test_ctx)}개  | 여행(1): {sum(test_labels)}개")

# ──────────────────────────────────────────
# 3-1. 클래스 가중치 계산 (학습 데이터 기준)
# ──────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
loss_fn = CrossEntropyLoss(weight=class_weights)
print(f"클래스 가중치: 일반(0)={class_weights[0]:.4f} / 여행(1)={class_weights[1]:.4f}")

# ──────────────────────────────────────────
# 4. 토크나이저 & 데이터셋
# ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

class ContextDataset(Dataset):
    def __init__(self, contexts, labels):
        self.encodings = tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=384,
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

train_dataset = ContextDataset(train_ctx,  train_labels)
val_dataset   = ContextDataset(val_ctx,    val_labels)
test_dataset  = ContextDataset(test_ctx,   test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# ──────────────────────────────────────────
# 5. 모델
# ──────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/roberta-base",
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = loss_fn.to(device)
print(f"\n사용 디바이스: {device}")

# ──────────────────────────────────────────
# 6. 옵티마이저 + 스케줄러
# ──────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

total_steps  = len(train_loader) * 100
warmup_steps = len(train_loader)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ──────────────────────────────────────────
# 7. 얼리스탑
# ──────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
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
# 8. 학습 (최대 100 에폭)
# ──────────────────────────────────────────
train_losses = []
val_losses   = []
val_accs     = []
best_val_acc = 0

for epoch in range(100):
    # 학습
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(outputs.logits, labels_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

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
            labels_batch   = batch["labels"].to(device)

            outputs  = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss     = loss_fn(outputs.logits, labels_batch)
            val_loss += loss.item()
            preds     = torch.argmax(outputs.logits, dim=1)
            correct  += (preds == labels_batch).sum().item()
            total    += len(labels_batch)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = correct / total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("./model1_relabeled_window3_best")
        tokenizer.save_pretrained("./model1_relabeled_window3_best")
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} ← 최고 저장!")
    else:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    early_stopping(val_acc)
    if early_stopping.stop:
        print(f"\n얼리스탑! {epoch+1} 에폭에서 중단")
        print(f"최고 검증 정확도: {best_val_acc:.4f}")
        break

# ──────────────────────────────────────────
# 9. 테스트
# ──────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained("./model1_relabeled_window3_best")
model.to(device)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

print("\n=== 테스트 결과 ===")
print(classification_report(
    all_labels, all_preds,
    target_names=["일반(0)", "여행(1)"],
    zero_division=0
))

# ──────────────────────────────────────────
# 10. 시각화
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3)
axes[0].plot(val_losses,   label='Val Loss',   marker='o', markersize=3)
axes[0].set_title('Loss 변화')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(val_accs, label='Val Accuracy',
             marker='o', markersize=3, color='green')
axes[1].axhline(y=best_val_acc, color='red',
                linestyle='--', label=f'최고: {best_val_acc:.4f}')
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
plt.savefig('model1_relabeled_window3_results.png', dpi=150)
plt.show()
print("시각화 저장 완료: model1_relabeled_window3_results.png")

# ──────────────────────────────────────────
# 11. 실제 문장 테스트
# ──────────────────────────────────────────
def predict(curr_text, prev_texts=None, next_texts=None):
    parts = []
    if prev_texts:
        parts.extend(prev_texts)
    parts.append(curr_text)
    if next_texts:
        parts.extend(next_texts)

    context = " [SEP] ".join(parts)
    inputs  = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred    = torch.argmax(outputs.logits, dim=1).item()
        prob    = torch.softmax(outputs.logits, dim=1)[0]

    label = "여행" if pred == 1 else "일반"
    print(f"문장: {curr_text}")
    print(f"앞:   {prev_texts}")
    print(f"뒤:   {next_texts}")
    print(f"예측: {label} (여행 확률: {prob[1]:.2%})")
    print()

print("\n=== 실제 문장 테스트 ===")

# 여행 문맥 (window=3)
predict(
    "오 좋다 언제 갈까",
    prev_texts=["제주도 여행 어때?", "제주도 가자!", "맞아 좋지"],
    next_texts=["이번 주말 어때?", "나 토요일 가능", "나도 토요일 괜찮아"]
)

# 일상 문맥 (window=3)
predict(
    "좋아!",
    prev_texts=["배고프다", "뭐 먹을까?", "치킨 먹을까?"],
    next_texts=["나도 배고파", "배달 시키자", "뭐 시킬까?"]
)

# 여행 문맥에서의 좋아! (window=3)
predict(
    "좋아!",
    prev_texts=["이번 주말 뭐 해?", "제주도 가자!", "언제 갈까?"],
    next_texts=["언제 갈까?", "이번 주말 어때?", "나 토요일 가능"]
)

# 여행 중 식사 계획 (window=3)
predict(
    "칠돈가 가자",
    prev_texts=["제주도 도착했다", "숙소 체크인했어", "저녁 뭐 먹을까?"],
    next_texts=["좋아!", "거기 유명하잖아", "몇 시에 갈까?"]
)

# 일상 대화 (window=3)
predict(
    "나 너무 피곤해",
    prev_texts=["과제 언제 해?", "나도 바빠", "오늘 수업 많았어?"],
    next_texts=["나도 힘들어", "그냥 자자", "내일 또 수업 있어?"]
)