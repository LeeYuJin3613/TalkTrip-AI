"""
Stage 3: Intent 분류기 (7-way classification)

의도 클래스 (7개):
    PROPOSE, AGREE, DISAGREE, CONFIRM, CANCEL, QUERY, OTHER

핵심 이슈: DISAGREE 클래스 극심한 불균형 (전체 1.9%)
해결: Weighted CrossEntropyLoss (역빈도 가중치)

CANCEL은 학습 데이터 0개 → 모델이 학습 못함 (정상)
classification_report에서 labels= 명시해서 7개 클래스 모두 출력 강제.

핵심 디자인:
- 입력: "{ctx_3} [SEP] {ctx_2} [SEP] {ctx_1} [SEP] {target}"
   (stage1과 동일한 context window. "오케이"가 PROPOSE 다음이면 AGREE)
- 모델: KLUE-RoBERTa + 7-way classification head
- 평가 지표: Macro F1 (accuracy는 불균형에 둔감)

사용법:
    pip install torch transformers scikit-learn tqdm
    python stage3.py --data_file ./dataset/final.json
"""

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score


MODEL_NAME = 'klue/roberta-base'
MAX_LENGTH = 256

INTENTS = ['PROPOSE', 'AGREE', 'DISAGREE', 'CONFIRM', 'CANCEL', 'QUERY', 'OTHER']
INTENT2ID = {x: i for i, x in enumerate(INTENTS)}
ID2INTENT = {i: x for x, i in INTENT2ID.items()}
ALL_LABEL_IDS = list(range(len(INTENTS)))  # [0..6] - 모든 클래스 강제 표시용


# ============================================================
# 데이터 준비
# ============================================================

def load_data(data_file):
    """라벨 JSON 로드. dict({'messages': [...]}) 또는 list 형식 모두 지원."""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    
    with open(path, encoding='utf-8') as fp:
        data = json.load(fp)
    
    if isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError("'messages' 키를 가진 dict이거나 list여야 함")
    
    print(f"  {path.name}: {len(messages)}개 메시지 로드")
    messages.sort(key=lambda m: m['global_idx'])
    return messages


def add_context(messages, num_prev=3):
    """직전 N개 메시지를 context로 추가 (같은 채팅 내에서만)"""
    for i, m in enumerate(messages):
        ctx = []
        for j in range(max(0, i - num_prev), i):
            if messages[j]['source'] == m['source']:
                ctx.append(messages[j]['text'])
        m['context'] = ctx
    return messages


def split_by_chat(messages, train_ratio=0.7, val_ratio=0.15, seed=42):
    """채팅(source) 단위 분할 - leakage 방지"""
    sources = sorted(set(m['source'] for m in messages))
    random.seed(seed)
    random.shuffle(sources)
    
    n = len(sources)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_sources = set(sources[:n_train])
    val_sources = set(sources[n_train:n_train + n_val])
    test_sources = set(sources[n_train + n_val:])
    
    return {
        'train': [m for m in messages if m['source'] in train_sources],
        'val':   [m for m in messages if m['source'] in val_sources],
        'test':  [m for m in messages if m['source'] in test_sources],
    }


def print_data_stats(splits):
    for name in ['train', 'val', 'test']:
        msgs = splits[name]
        n_total = len(msgs)
        n_chats = len(set(m['source'] for m in msgs))
        intents = Counter(m['intent_primary'] for m in msgs
                          if m.get('intent_primary') in INTENT2ID)
        print(f"  [{name:5s}] 채팅 {n_chats}개, 메시지 {n_total}개")
        print(f"          의도 분포: {dict(intents)}")


# ============================================================
# Class weight 계산
# ============================================================

def compute_class_weights(messages):
    """역빈도 가중치 (sklearn class_weight='balanced'와 동일)"""
    counts = Counter(m['intent_primary'] for m in messages
                     if m.get('intent_primary') in INTENT2ID)
    total = sum(counts.values())
    n_classes = len(INTENTS)
    weights = []
    for intent in INTENTS:
        c = counts.get(intent, 1)  # division-by-zero 방지
        weights.append(total / (n_classes * c))
    return torch.tensor(weights, dtype=torch.float), counts


# ============================================================
# Dataset
# ============================================================

class IntentDataset(Dataset):
    def __init__(self, messages, tokenizer, max_length=MAX_LENGTH):
        self.messages = [m for m in messages 
                         if m.get('intent_primary') in INTENT2ID]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        m = self.messages[idx]
        ctx = m.get('context', [])
        if ctx:
            text = ' [SEP] '.join(ctx) + ' [SEP] ' + m['text']
        else:
            text = m['text']
        
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(INTENT2ID[m['intent_primary']], dtype=torch.long),
        }


# ============================================================
# 평가
# ============================================================

def evaluate(model, loader, device, desc='Eval'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            labels.extend(batch['labels'].numpy().tolist())
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            preds.extend(outputs.logits.argmax(-1).cpu().numpy().tolist())
    return preds, labels


def safe_classification_report(labels, preds):
    """모든 7개 클래스 표시 (val에 없는 클래스도 0으로 표시)"""
    return classification_report(
        labels, preds,
        labels=ALL_LABEL_IDS,
        target_names=INTENTS,
        digits=3,
        zero_division=0,
    )


def safe_macro_f1(labels, preds):
    """val에 없는 클래스도 포함한 macro F1 (정직한 평가)"""
    return f1_score(
        labels, preds,
        average='macro',
        labels=ALL_LABEL_IDS,
        zero_division=0,
    )


# ============================================================
# 학습
# ============================================================

def train(args):
    print("\n=== 1. 데이터 로드 ===")
    messages = load_data(args.data_file)
    messages = add_context(messages, num_prev=args.num_prev)
    splits = split_by_chat(messages, seed=args.seed)
    
    print("\n=== 2. 데이터 분할 ===")
    print_data_stats(splits)
    
    print(f"\n=== 3. 모델 로딩 ({MODEL_NAME}) ===")
    print("  (첫 실행 시 ~440MB 다운로드)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENTS),
        id2label=ID2INTENT,
        label2id=INTENT2ID,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device      : {device}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Classes     : {len(INTENTS)}-way ({', '.join(INTENTS)})")
    
    class_weights, counts = compute_class_weights(splits['train'])
    print(f"\n=== Class 분포 및 가중치 ===")
    for intent in INTENTS:
        c = counts.get(intent, 0)
        w = class_weights[INTENT2ID[intent]].item()
        marker = " ⚠ 학습 데이터 없음" if c == 0 else ""
        print(f"    {intent:10s}: {c:4d}개  → weight {w:.2f}{marker}")
    
    train_ds = IntentDataset(splits['train'], tokenizer)
    val_ds = IntentDataset(splits['val'], tokenizer)
    print(f"\n  Intent train: {len(train_ds)}")
    print(f"  Intent val  : {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    os.makedirs(args.output, exist_ok=True)
    history = []
    best_f1 = 0
    start_time = time.time()
    
    print(f"\n=== 4. 학습 시작 (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}) ===\n")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            loss = loss_fn(outputs.logits, batch['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        preds, labels = evaluate(model, val_loader, device, desc='Val')
        macro_f1 = safe_macro_f1(labels, preds)
        elapsed = time.time() - start_time
        
        # === 체크포인트 먼저 저장 (리포트가 깨져도 모델은 보존) ===
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_macro_f1': macro_f1,
        })
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            saved = True
        else:
            saved = False
        
        # === 그 다음 리포트 출력 ===
        print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}  val_macro_f1={macro_f1:.4f}  ({elapsed:.0f}s)")
        print(safe_classification_report(labels, preds))
        
        if saved:
            print(f"  ✓ Best Macro F1 갱신 → {args.output}/ 에 저장")
        else:
            print(f"  · 이전 best F1 ({best_f1:.4f}) 보다 낮음, 저장 안 함")
    
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if splits['test']:
        print(f"\n=== 5. Test 평가 (best checkpoint) ===")
        model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
        test_ds = IntentDataset(splits['test'], tokenizer)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        preds, labels = evaluate(model, test_loader, device, desc='Test')
        test_f1 = safe_macro_f1(labels, preds)
        print(f"\nTest Macro F1: {test_f1:.4f}")
        print(safe_classification_report(labels, preds))
    
    total_time = time.time() - start_time
    print(f"\n=== 완료 ===")
    print(f"총 학습 시간      : {total_time/60:.1f}분")
    print(f"최종 best val Macro F1: {best_f1:.4f}")
    print(f"체크포인트        : {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 3 Intent 분류기 학습')
    parser.add_argument('--data_file', required=True,
                        help='라벨 JSON 파일 경로 (예: ./dataset/final.json)')
    parser.add_argument('--output', default='./checkpoints/stage3',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Intent 분류는 클래스 많아 더 많은 epoch 필요')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_prev', type=int, default=3,
                        help='context로 쓸 직전 메시지 개수')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42,
                        help='stage1과 같은 seed 권장 (같은 분할 보장)')
    args = parser.parse_args()
    train(args)