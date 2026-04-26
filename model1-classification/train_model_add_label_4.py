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
from collections import defaultdict, Counter

# ──────────────────────────────────────────
# 설정값 (여기만 수정하면 됨)
# ──────────────────────────────────────────
DATA_PATH   = r"C:\dev\TalkTrip\TripTalk_ai_model\dataset\test-add-aihub-relabeled.json"
SAVE_DIR    = "./model1_v6_dropout04_lr5e6"
MODEL_NAME  = "klue/roberta-base"
NUM_LABELS  = 4
MAX_LENGTH  = 384
BATCH_SIZE  = 16
MAX_EPOCHS  = 100
LR          = 5e-6
WINDOW      = 3
PATIENCE    = 5
SEED        = 42

# 클래스 가중치 (역빈도 기반)
# {0: 2281, 1: 1632, 2: 765, 3: 695} 기준
CLASS_WEIGHTS = [0.59, 0.82, 1.76, 1.93]

# confidence 필터: "high"만 학습에 사용
USE_CONFIDENCE_FILTER = True

# ──────────────────────────────────────────
# 재현성 고정
# ──────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────
# 라벨 통계 출력 헬퍼
# ──────────────────────────────────────────
def label_stats(labels):
    c = Counter(labels)
    return " | ".join([f"라벨{k}: {v}개" for k, v in sorted(c.items())])

# ──────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# None / 문자열 라벨 제거 + int 변환
raw_data = [d for d in raw_data if d.get("label") is not None]
for d in raw_data:
    d["label"] = int(d["label"])

# 유효 라벨 범위 검사 (에러 방지)
valid_labels = set(range(NUM_LABELS))
before = len(raw_data)
raw_data = [d for d in raw_data if d["label"] in valid_labels]
if len(raw_data) != before:
    print(f"⚠️  범위 밖 라벨 제거: {before}개 → {len(raw_data)}개 (제거: {before - len(raw_data)}개)")

# confidence 필터
if USE_CONFIDENCE_FILTER:
    before = len(raw_data)
    raw_data = [d for d in raw_data if d.get("confidence", "high") == "high"]
    print(f"confidence 필터: {before}개 → {len(raw_data)}개 (제외: {before - len(raw_data)}개)")

# source 기준 그룹핑
file_groups = defaultdict(list)
for item in raw_data:
    file_groups[item["source"]].append(item)

print(f"전체 파일 수: {len(file_groups)}개 | 전체 문장: {len(raw_data)}개")
print(f"라벨 분포: {dict(sorted(Counter(d['label'] for d in raw_data).items()))}")

# ──────────────────────────────────────────
# 2. 슬라이딩 윈도우
# ──────────────────────────────────────────
def make_split_data(sources, file_groups, window=WINDOW):
    contexts = []
    labels   = []

    for source in sources:
        items = file_groups[source]
        for i, item in enumerate(items):
            prev_texts = [items[j]["text"] for j in range(max(0, i - window), i)]
            curr_text  = item["text"]
            next_texts = [items[j]["text"] for j in range(i + 1, min(len(items), i + window + 1))]
            context    = " [SEP] ".join(prev_texts + [curr_text] + next_texts)
            contexts.append(context)
            labels.append(item["label"])

    return contexts, labels

# ──────────────────────────────────────────
# 3. 파일 단위 train / val / test 분리
# ──────────────────────────────────────────
all_sources = list(file_groups.keys())
random.shuffle(all_sources)

n             = len(all_sources)
train_sources = all_sources[:int(n * 0.70)]
val_sources   = all_sources[int(n * 0.70):int(n * 0.85)]
test_sources  = all_sources[int(n * 0.85):]

train_ctx, train_labels = make_split_data(train_sources, file_groups)
val_ctx,   val_labels   = make_split_data(val_sources,   file_groups)
test_ctx,  test_labels  = make_split_data(test_sources,  file_groups)

# 수정: sum() 대신 Counter로 정확한 분포 출력
print(f"\n학습  파일: {len(train_sources):3d}개 | 문장: {len(train_ctx):5d}개 | {label_stats(train_labels)}")
print(f"검증  파일: {len(val_sources):3d}개   | 문장: {len(val_ctx):5d}개   | {label_stats(val_labels)}")
print(f"테스트 파일: {len(test_sources):3d}개 | 문장: {len(test_ctx):5d}개  | {label_stats(test_labels)}")

# ──────────────────────────────────────────
# 4. 토크나이저 & 데이터셋
# ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ContextDataset(Dataset):
    def __init__(self, contexts, labels):
        self.encodings = tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

train_dataset = ContextDataset(train_ctx, train_labels)
val_dataset   = ContextDataset(val_ctx,   val_labels)
test_dataset  = ContextDataset(test_ctx,  test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ──────────────────────────────────────────
# 5. 모델
# ──────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    hidden_dropout_prob=0.4,
    attention_probs_dropout_prob=0.4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\n사용 디바이스: {device}")

# 클래스 가중치 적용
weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(device)
loss_fn = CrossEntropyLoss(weight=weights)

# ──────────────────────────────────────────
# 6. 옵티마이저 + 스케줄러
# ──────────────────────────────────────────
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps  = len(train_loader) * MAX_EPOCHS
warmup_steps = len(train_loader)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ──────────────────────────────────────────
# 7. 얼리스탑 (Val Loss 기준)
# ──────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            print(f"  얼리스탑 카운터: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_loss
            self.counter    = 0

early_stopping = EarlyStopping(patience=PATIENCE)

# ──────────────────────────────────────────
# 8. 학습 루프
# ──────────────────────────────────────────
train_losses  = []
val_losses    = []
val_accs      = []
best_val_loss = float("inf")

for epoch in range(MAX_EPOCHS):
    # ── 학습 ──
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── 검증 ──
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss     = loss_fn(outputs.logits, labels_batch)
            val_loss += loss.item()
            preds     = torch.argmax(outputs.logits, dim=1)
            correct  += (preds == labels_batch).sum().item()
            total    += len(labels_batch)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = correct / total
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} ← 최고 저장!")
    else:
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    early_stopping(avg_val_loss)
    if early_stopping.stop:
        print(f"\n얼리스탑! {epoch+1} 에폭에서 중단")
        print(f"최고 Val Loss: {best_val_loss:.4f}")
        break

# ──────────────────────────────────────────
# 9. 테스트
# ──────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
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

target_names = ["일반(0)", "장소(1)", "시간(2)", "일정(3)"] if NUM_LABELS == 4 else ["일반(0)", "여행(1)"]

print("\n=== 테스트 결과 ===")
print(classification_report(
    all_labels, all_preds,
    target_names=target_names,
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

best_val_acc = max(val_accs)
axes[1].plot(val_accs, label='Val Accuracy', marker='o', markersize=3, color='green')
axes[1].axhline(y=best_val_acc, color='red', linestyle='--',
                label=f'최고: {best_val_acc:.4f}')
axes[1].set_title('검증 정확도 변화')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim([0, 1])
axes[1].legend()
axes[1].grid(True)

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names,
    ax=axes[2]
)
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('예측')
axes[2].set_ylabel('실제')

plt.tight_layout()
img_path = f'{SAVE_DIR.replace("./", "")}_results.png'
plt.savefig(img_path, dpi=150)
plt.show()
print(f"시각화 저장 완료: {img_path}")

# ──────────────────────────────────────────
# 11. 실제 문장 테스트
# ──────────────────────────────────────────
def predict(curr_text, prev_texts=None, next_texts=None):
    parts   = (prev_texts or []) + [curr_text] + (next_texts or [])
    context = " [SEP] ".join(parts)
    inputs  = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        pred    = torch.argmax(outputs.logits, dim=1).item()
        prob    = torch.softmax(outputs.logits, dim=1)[0]

    label = target_names[pred]
    print(f"문장: {curr_text}")
    print(f"앞:   {prev_texts}")
    print(f"뒤:   {next_texts}")
    print(f"예측: {label} | 확률: {[f'{p:.2%}' for p in prob.tolist()]}")
    print()

print("\n=== 실제 문장 테스트 ===")

predict(
    "오 좋다 언제 갈까",
    prev_texts=["제주도 여행 어때?", "제주도 가자!", "맞아 좋지"],
    next_texts=["이번 주말 어때?", "나 토요일 가능", "나도 토요일 괜찮아"]
)

predict(
    "좋아!",
    prev_texts=["배고프다", "뭐 먹을까?", "치킨 먹을까?"],
    next_texts=["나도 배고파", "배달 시키자", "뭐 시킬까?"]
)

predict(
    "좋아!",
    prev_texts=["이번 주말 뭐 해?", "제주도 가자!", "언제 갈까?"],
    next_texts=["언제 갈까?", "이번 주말 어때?", "나 토요일 가능"]
)

predict(
    "칠돈가 가자",
    prev_texts=["제주도 도착했다", "숙소 체크인했어", "저녁 뭐 먹을까?"],
    next_texts=["좋아!", "거기 유명하잖아", "몇 시에 갈까?"]
)

predict(
    "나 너무 피곤해",
    prev_texts=["과제 언제 해?", "나도 바빠", "오늘 수업 많았어?"],
    next_texts=["나도 힘들어", "그냥 자자", "내일 또 수업 있어?"]
)

# 여행 맥락
predict(
    "저녁은 어디서 먹을래?",
    prev_texts=["숙소 체크인 완료!", "오동도 산책 다 했어", "슬슬 배고프다"],
    next_texts=["낭만포차 거리 어때?", "거기 돌문어삼합 맛있다던데", "걸어서 10분이면 가"]
)

# 비여행 맥락
predict(
    "저녁은 어디서 먹을래?",
    prev_texts=["과제 언제 끝나?", "나 오늘 알바 6시에 끝나", "롤 랭겜 한 판 하고 싶은데"],
    next_texts=["그냥 편의점 가자", "나 돈 없어서 집에서 먹을 듯", "다음에 같이 먹자"]
)