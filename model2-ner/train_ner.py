"""
=============================================================================
Korean BERT NER 모델 학습 스크립트 (v2 — 체크포인트 & 얼리스탑)
- 엔티티: 날짜(DT), 장소(LC), 시간(TI)
- Colab TPU v5e / GPU(RTX 5070) 자동 감지
- ✅ 체크포인트: 매 에폭 저장, Colab 끊겨도 이어서 학습
- ✅ 얼리스탑: Val Loss 개선 없으면 자동 정지
=============================================================================

[사용법]
  # 처음 학습
  python train_ner.py

  # Colab 끊긴 후 이어서 학습 (자동 감지)
  python train_ner.py

  # 체크포인트 무시하고 처음부터
  python train_ner.py --reset
=============================================================================
"""

import json
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,                    # ← 변경됨
    AutoModelForTokenClassification,  # ← 변경됨
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import (
    classification_report,
    f1_score as seq_f1_score,
    precision_score as seq_precision_score,
    recall_score as seq_recall_score,
)
from seqeval.scheme import IOB2
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정 (필요에 따라 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 42
MODEL_NAME = "klue/bert-base"           # 한국어 BERT 사전학습 모델
MAX_LEN = 256                            # 최대 시퀀스 길이
BATCH_SIZE = 32                          # GPU 12GB
LEARNING_RATE = 3e-5
NUM_EPOCHS = 100
WARMUP_RATIO = 0.1
CONTEXT_WINDOW = 2                       # 앞뒤 N개 문장을 맥락으로 포함 (0=단일 문장)
DATA_PATH = "bert_ner.json"             # 학습 데이터 경로
SAVE_DIR = "ner_model_output"           # Best 모델 저장 디렉토리
CHECKPOINT_DIR = "ner_checkpoints"      # 체크포인트 디렉토리

# 얼리스탑 설정
EARLY_STOP_PATIENCE = 12               # Val Loss가 N 에폭 연속 개선 안 되면 정지
EARLY_STOP_MIN_DELTA = 0.001             # 이 값 이상 줄어야 "개선"으로 인정

# BIO 라벨 정의
LABEL_LIST = ["O", "B-DT", "I-DT", "B-LC", "I-LC", "B-TI", "I-TI"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """TPU / GPU / CPU 자동 감지"""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"✅ TPU 감지: {device}")
        return device, "tpu"
    except (ImportError, RuntimeError):
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU 감지: {name} ({mem:.1f} GB)")
        return device, "gpu"

    print("⚠️  CPU 사용 (학습이 매우 느릴 수 있습니다)")
    return torch.device("cpu"), "cpu"


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"📂 전체 데이터 로드: {len(raw)}개")
    return raw


def split_data(data, test_size=0.15, val_size=0.15, seed=42):
    """Train(70%) / Val(15%) / Test(15%) 분할"""
    train_val, test = train_test_split(data, test_size=test_size, random_state=seed)
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed)
    print(f"   Train: {len(train)}개 / Val: {len(val)}개 / Test: {len(test)}개")
    return train, val, test


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  체크포인트 저장 & 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def save_checkpoint(
    model, optimizer, scheduler, epoch,
    best_val_loss, best_val_f1, best_epoch,
    no_improve, history, checkpoint_dir
):
    """
    매 에폭마다 전체 학습 상태 저장
    Colab이 끊겨도 여기서부터 이어서 학습 가능
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "history": history,
        "seed": SEED,
    }

    # 임시 파일로 저장 후 rename (중간에 끊겨도 파일 손상 방지)
    tmp_path = ckpt_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)

    # 히스토리도 JSON으로 별도 저장 (확인 편의)
    with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return ckpt_path


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    """
    체크포인트가 있으면 로드하고 이어서 학습
    없으면 None 반환
    """
    ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    if not os.path.exists(ckpt_path):
        return None

    print(f"\n🔄 체크포인트 발견: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    info = {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "best_val_f1": checkpoint["best_val_f1"],
        "best_epoch": checkpoint["best_epoch"],
        "no_improve": checkpoint["no_improve"],
        "history": checkpoint["history"],
    }

    print(f"   ✅ Epoch {info['epoch']}까지 학습 완료 상태 복원")
    print(f"   Best Val Loss: {info['best_val_loss']:.4f} (Epoch {info['best_epoch']})")
    print(f"   Best Val F1:   {info['best_val_f1']:.4f}")
    print(f"   남은 에폭: {NUM_EPOCHS - info['epoch']}개")

    return info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  얼리스탑 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EarlyStopping:
    """
    Val Loss 기준 얼리스탑
    - patience 에폭 동안 val_loss가 min_delta 이상 줄어들지 않으면 정지
    - 체크포인트에서 복원 가능하도록 상태를 딕셔너리로 관리
    """

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        """에폭마다 호출. True 반환 시 학습 중지"""
        if val_loss < self.best_loss - self.min_delta:
            # 개선됨
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 개선 안 됨
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_state(self, best_loss, counter):
        """체크포인트에서 복원"""
        self.best_loss = best_loss
        self.counter = counter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  데이터셋 클래스 (컨텍스트 윈도우 지원)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NERDataset(Dataset):
    """
    컨텍스트 윈도우를 적용한 NER 데이터셋

    window=2 일 때 입력 구조:
      [CLS] 앞문장2 [SEP] 앞문장1 [SEP] ★타겟문장★ [SEP] 뒷문장1 [SEP] 뒷문장2 [SEP]

    - 타겟 문장의 토큰만 실제 BIO 라벨 부여
    - 맥락 문장의 토큰은 -100 (loss 계산에서 제외, 어텐션은 참여)
    - window=0 이면 기존과 동일하게 단일 문장만 사용
    """

    def __init__(self, data, tokenizer, max_len, label2id, context_window=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.window = context_window

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ── 타겟 문장 ──
        target = self.data[idx]
        target_text = "".join(target["tokens"])
        target_labels = target["labels"]

        if self.window == 0:
            # 단일 문장 모드 (기존과 동일)
            return self._encode_single(target_text, target_labels)

        # ── 컨텍스트 윈도우 구성 ──
        # 앞뒤 window개 문장을 가져와 하나의 텍스트로 합침
        # 구분자 " " 사용 (BERT가 자체적으로 서브워드 분리)
        SEP = " "
        parts = []  # (text, is_target)

        # 앞 문장들 (먼 것부터 → 가까운 것 순서)
        for i in range(idx - self.window, idx):
            if 0 <= i < len(self.data):
                parts.append(("".join(self.data[i]["tokens"]), False))

        # 타겟 문장
        target_part_idx = len(parts)
        parts.append((target_text, True))

        # 뒷 문장들
        for i in range(idx + 1, idx + self.window + 1):
            if 0 <= i < len(self.data):
                parts.append(("".join(self.data[i]["tokens"]), False))

        # 전체 텍스트 조합 & 타겟 위치 추적
        full_text = ""
        target_char_start = -1
        target_char_end = -1

        for j, (text, is_target) in enumerate(parts):
            if j > 0:
                full_text += SEP

            if is_target:
                target_char_start = len(full_text)

            full_text += text

            if is_target:
                target_char_end = len(full_text)

        # ── BERT 토큰화 ──
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # ── 라벨 매핑 ──
        # 타겟 문장 범위 내 토큰만 실제 라벨, 나머지는 -100
        labels = []
        for start, end in offset_mapping:
            s, e = start.item(), end.item()

            if s == 0 and e == 0:
                # 특수 토큰 ([CLS], [SEP], [PAD])
                labels.append(-100)
            elif s >= target_char_start and e <= target_char_end:
                # ★ 타겟 문장 내 토큰 → 실제 라벨 부여
                char_idx = s - target_char_start
                if char_idx < len(target_labels):
                    labels.append(self.label2id.get(target_labels[char_idx], 0))
                else:
                    labels.append(-100)
            else:
                # 맥락 문장 토큰 → loss 제외 (어텐션은 참여)
                labels.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _encode_single(self, text, char_labels):
        """window=0 일 때 단일 문장 인코딩 (기존 로직)"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        labels = []
        for start, end in offset_mapping:
            s, e = start.item(), end.item()
            if s == 0 and e == 0:
                labels.append(-100)
            elif s < len(char_labels):
                labels.append(self.label2id.get(char_labels[s], 0))
            else:
                labels.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  학습 & 평가 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_epoch(model, dataloader, optimizer, scheduler, device, device_type):
    model.train()
    total_loss = 0
    steps = 0

    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss = outputs.loss
        total_loss += loss.item()
        steps += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if device_type == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()

    return total_loss / max(steps, 1)


def evaluate(model, dataloader, device, device_type):
    """seqeval 기반 엔티티 단위 정밀 평가"""
    model.eval()
    total_loss = 0
    steps = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += outputs.loss.item()
            steps += 1

            preds = torch.argmax(outputs.logits, dim=-1)

            for i in range(preds.shape[0]):
                pred_seq, label_seq = [], []
                for j in range(preds.shape[1]):
                    lbl = lbls[i][j].item()
                    if lbl != -100:
                        pred_seq.append(ID2LABEL[preds[i][j].item()])
                        label_seq.append(ID2LABEL[lbl])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    avg_loss = total_loss / max(steps, 1)
    f1 = seq_f1_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    prec = seq_precision_score(all_labels, all_preds, mode="strict", scheme=IOB2)
    rec = seq_recall_score(all_labels, all_preds, mode="strict", scheme=IOB2)

    return avg_loss, f1, prec, rec, all_labels, all_preds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    reset_mode = "--reset" in sys.argv
    set_seed(SEED)
    device, device_type = get_device()

    print("\n" + "=" * 65)
    print("  🚀 Korean BERT NER 학습 (v2 — 체크포인트 & 얼리스탑)")
    print(f"  모델: {MODEL_NAME}")
    print(f"  설정: MAX_LEN={MAX_LEN}, BATCH={BATCH_SIZE}, LR={LEARNING_RATE}")
    print(f"  에폭: {NUM_EPOCHS}, 디바이스: {device_type.upper()}")
    print(f"  컨텍스트 윈도우: {CONTEXT_WINDOW} (앞뒤 {CONTEXT_WINDOW}문장 = 총 {CONTEXT_WINDOW*2+1}문장 입력)")
    print(f"  얼리스탑: patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"  체크포인트: {CHECKPOINT_DIR}/")
    print("=" * 65)

    # ── 리셋 모드 ──
    if reset_mode:
        import shutil
        if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
            print("🗑️  체크포인트 초기화 완료 (처음부터 학습)")

    # ── 데이터 로드 & 분할 ──
    data = load_data(DATA_PATH)
    train_data, val_data, test_data = split_data(data)

    # ── 토크나이저 & 모델 ──
    print(f"\n🔧 토크나이저 로드중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # ← AutoTokenizer

    print(f"🔧 모델 로드중: {MODEL_NAME} (num_labels={NUM_LABELS})")
    model = AutoModelForTokenClassification.from_pretrained(  # ← AutoModelForTokenClassification
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)  # ← 이 줄은 그대로 두세요

    # ── 데이터로더 ──
    train_ds = NERDataset(train_data, tokenizer, MAX_LEN, LABEL2ID, context_window=CONTEXT_WINDOW)
    val_ds = NERDataset(val_data, tokenizer, MAX_LEN, LABEL2ID, context_window=CONTEXT_WINDOW)
    test_ds = NERDataset(test_data, tokenizer, MAX_LEN, LABEL2ID, context_window=CONTEXT_WINDOW)

    pin = device_type == "gpu"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=pin)

    # ── 옵티마이저 & 스케줄러 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── 체크포인트 복원 시도 ──
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    history = []

    ckpt_info = load_checkpoint(CHECKPOINT_DIR, model, optimizer, scheduler, device)
    if ckpt_info is not None:
        start_epoch = ckpt_info["epoch"] + 1
        best_val_loss = ckpt_info["best_val_loss"]
        best_val_f1 = ckpt_info["best_val_f1"]
        best_epoch = ckpt_info["best_epoch"]
        no_improve = ckpt_info["no_improve"]
        history = ckpt_info["history"]

        if start_epoch > NUM_EPOCHS:
            print(f"\n✅ 이미 {NUM_EPOCHS} 에폭 학습 완료. 테스트만 진행합니다.")
    else:
        print("\n📝 체크포인트 없음 — 처음부터 학습 시작")

    # ── 얼리스탑 초기화 ──
    early_stopper = EarlyStopping(
        patience=EARLY_STOP_PATIENCE,
        min_delta=EARLY_STOP_MIN_DELTA,
    )
    if ckpt_info is not None:
        early_stopper.restore_state(best_val_loss, no_improve)

    # ── 학습 루프 ──
    if start_epoch <= NUM_EPOCHS:
        header = f"{'Epoch':>5} │ {'Train Loss':>10} │ {'Val Loss':>8} │ {'Val F1':>7} │ {'Val P':>6} │ {'Val R':>6} │ {'Time':>8}"
        print(f"\n{header}")
        print("─" * len(header))

        # 이전 히스토리 요약 출력 (이어서 학습 시)
        if history:
            for h in history:
                mark = " ★" if h["epoch"] == best_epoch else ""
                print(
                    f"{h['epoch']:5d} │ {h['train_loss']:10.4f} │ {h['val_loss']:8.4f} │ "
                    f"{h['val_f1']:7.4f} │ {h['val_precision']:6.4f} │ {h['val_recall']:6.4f} │ {'(복원)':>8}{mark}"
                )
            print("─" * len(header))

        total_start = time.time()

        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            ep_start = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, device_type)
            val_loss, val_f1, val_p, val_r, _, _ = evaluate(model, val_loader, device, device_type)

            ep_time = timedelta(seconds=int(time.time() - ep_start))

            # Best 모델 갱신 (F1 기준)
            marker = ""
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                os.makedirs(SAVE_DIR, exist_ok=True)
                model.save_pretrained(SAVE_DIR)
                tokenizer.save_pretrained(SAVE_DIR)
                marker = " ★ best"
                print(f"        🏆 Best 모델 업데이트 (Epoch {epoch})")

            # Val Loss 개선 체크 (얼리스탑용)
            if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

            print(
                f"{epoch:5d} │ {train_loss:10.4f} │ {val_loss:8.4f} │ {val_f1:7.4f} │ {val_p:6.4f} │ {val_r:6.4f} │ {str(ep_time):>8}{marker}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_f1": round(val_f1, 4),
                "val_precision": round(val_p, 4),
                "val_recall": round(val_r, 4),
            })

            # ── 체크포인트 저장 (매 에폭) ──
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_val_loss, best_val_f1, best_epoch,
                no_improve, history, CHECKPOINT_DIR,
            )
            print(f"        💾 체크포인트 저장: {ckpt_path}")

            # ── 얼리스탑 체크 ──
            if early_stopper.step(val_loss):
                print(f"\n⏹️  얼리스탑! Val Loss가 {EARLY_STOP_PATIENCE} 에폭 연속 개선되지 않음")
                print(f"   최저 Val Loss: {early_stopper.best_loss:.4f}")
                break

        total_time = timedelta(seconds=int(time.time() - total_start))
        print(f"\n⏱️  이번 세션 학습 시간: {total_time}")

        print(f"\n💾 최종 Best 모델 강제 저장 중... ({SAVE_DIR})")
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 모델 + 토크나이저 강제 저장
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        print(f"✅ 최종 모델 저장 완료 → {SAVE_DIR}")
        print(f"   (model.safetensors, config.json, tokenizer 파일 모두 포함)")

    print(f"\n🏆 Best: Epoch {best_epoch}, Val F1 = {best_val_f1:.4f}, Val Loss = {best_val_loss:.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 테스트 셋 최종 평가 (Best 모델)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 65)
    print("  📊 테스트 셋 최종 평가 (Best 모델)")
    print("=" * 65)

    if os.path.exists(SAVE_DIR):
        # ←←← 여기 수정됨
        model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR)
        model.to(device)
    else:
        print("  ⚠️  Best 모델이 없습니다. 현재 모델로 평가합니다.")

    test_loss, test_f1, test_p, test_r, test_labels, test_preds = evaluate(
        model, test_loader, device, device_type
    )

    print(f"\n  ┌───────────────────────────────────┐")
    print(f"  │  Test Loss      : {test_loss:.4f}          │")
    print(f"  │  Test F1-Score  : {test_f1:.4f}          │")
    print(f"  │  Test Precision : {test_p:.4f}          │")
    print(f"  │  Test Recall    : {test_r:.4f}          │")
    print(f"  └───────────────────────────────────┘")

    print("\n  === 엔티티별 상세 리포트 (seqeval, strict) ===\n")
    report = classification_report(
        test_labels, test_preds, mode="strict", scheme=IOB2, digits=4
    )
    print(report)

    # 결과 저장
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    test_result = {
        "test_loss": round(test_loss, 4),
        "test_f1": round(test_f1, 4),
        "test_precision": round(test_p, 4),
        "test_recall": round(test_r, 4),
        "best_epoch": best_epoch,
        "report": report,
    }
    with open(os.path.join(SAVE_DIR, "test_results.json"), "w") as f:
        json.dump(test_result, f, indent=2, ensure_ascii=False)

    print(f"\n📁 저장 완료:")
    print(f"   Best 모델      → {SAVE_DIR}/")
    print(f"   체크포인트      → {CHECKPOINT_DIR}/")
    print(f"   학습 히스토리   → {SAVE_DIR}/training_history.json")
    print(f"   테스트 결과     → {SAVE_DIR}/test_results.json")


if __name__ == "__main__":
    main()
