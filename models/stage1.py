"""
Stage 1: 여행 구간 이진 분류기 (단일 파일 버전)

데이터 로드 + 분할 + 학습 + 평가 모두 포함.

사용법:
    pip install torch transformers scikit-learn tqdm
    python stage1.py --data_file C:\\dev\\TalkTrip\\TripTalk_ai_model\\dataset\\final.json

옵션:
    --data_file      라벨 JSON 파일 경로 (1-20번 채팅 모두 포함)
    --output         체크포인트 저장 디렉토리 (기본: ./checkpoints/stage1)
    --epochs         epoch 수 (기본 5)
    --batch_size     배치 크기 (기본 16, OOM 나면 8)
    --lr             learning rate (기본 2e-5)
    --num_prev       context로 쓸 직전 메시지 개수 (기본 3)
    --seed           train/val/test 분할 시드 (기본 42)

데이터 형식 (final.json):
    {"messages": [{"id": "trip001_m00001", "global_idx": 1, "source": "trip_chat001",
                   "text": "...", "in_travel_span": true/false, ...}, ...]}
    또는 그냥 list 형식도 지원.
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
from sklearn.metrics import classification_report, f1_score, accuracy_score


MODEL_NAME = 'klue/roberta-base'
MAX_LENGTH = 256


# ============================================================
# 데이터 준비
# ============================================================

def load_data(data_file):
    """라벨 JSON 파일 로드. dict({'messages': [...]}) 또는 list 형식 모두 지원."""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {path}")
    
    with open(path, encoding='utf-8') as fp:
        data = json.load(fp)
    
    if isinstance(data, dict) and 'messages' in data:
        messages = data['messages']
    elif isinstance(data, list):
        messages = data
    else:
        raise ValueError(
            f"알 수 없는 JSON 구조. 'messages' 키를 가진 dict이거나 list여야 함. "
            f"실제 구조: {type(data).__name__}"
        )
    
    # 필수 필드 검증
    required = {'global_idx', 'source', 'text', 'in_travel_span'}
    missing = required - set(messages[0].keys())
    if missing:
        raise ValueError(f"메시지에 필수 필드 누락: {missing}")
    
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
    """채팅(source) 단위 분할 - 같은 대화 흐름이 split 간 섞이지 않도록"""
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
        n_in_span = sum(1 for m in msgs if m['in_travel_span'])
        n_chats = len(set(m['source'] for m in msgs))
        ratio = n_in_span / n_total * 100 if n_total else 0
        print(f"  [{name:5s}] 채팅 {n_chats}개, 메시지 {n_total}개, "
              f"in_span {n_in_span}개 ({ratio:.1f}%)")


# ============================================================
# Dataset
# ============================================================

class SpanDataset(Dataset):
    """
    각 메시지를 직전 N개 context와 함께 [SEP]로 연결해 입력.
    예: "야 어디 가자 [SEP] 제주 어때 [SEP] 오케이"
    """
    def __init__(self, messages, tokenizer, max_length=MAX_LENGTH):
        self.messages = messages
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
            'labels': torch.tensor(int(m['in_travel_span']), dtype=torch.long),
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
            outputs = model(**batch)
            preds.extend(outputs.logits.argmax(-1).cpu().numpy().tolist())
    return preds, labels


# ============================================================
# 학습
# ============================================================

def train(args):
    # --- 1. 데이터 준비 ---
    print("\n=== 1. 데이터 로드 ===")
    messages = load_data(args.data_file)
    messages = add_context(messages, num_prev=args.num_prev)
    splits = split_by_chat(messages, seed=args.seed)
    
    print("\n=== 2. 데이터 분할 ===")
    print_data_stats(splits)
    
    # --- 2. 모델/토크나이저 로딩 ---
    print(f"\n=== 3. 모델 로딩 ({MODEL_NAME}) ===")
    print("  (첫 실행 시 ~440MB 다운로드)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device      : {device}")
    print(f"  Parameters  : {n_params:,}")
    
    # --- 3. 데이터로더 ---
    train_ds = SpanDataset(splits['train'], tokenizer)
    val_ds = SpanDataset(splits['val'], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    
    # --- 4. Optimizer + Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    
    os.makedirs(args.output, exist_ok=True)
    history = []
    best_f1 = 0
    start_time = time.time()
    
    print(f"\n=== 4. 학습 시작 (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}) ===\n")
    
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        
        # --- Validation ---
        preds, labels = evaluate(model, val_loader, device, desc='Val')
        f1 = f1_score(labels, preds, pos_label=1)
        acc = accuracy_score(labels, preds)
        elapsed = time.time() - start_time
        
        print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}  val_acc={acc:.4f}  "
              f"val_f1={f1:.4f}  ({elapsed:.0f}s)")
        print(classification_report(
            labels, preds,
            target_names=['not_in_span', 'in_span'],
            digits=4,
            zero_division=0,
        ))
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_acc': acc,
            'val_f1': f1,
        })
        
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ✓ Best F1 갱신 → {args.output}/ 에 저장")
        else:
            print(f"  · 이전 best F1 ({best_f1:.4f}) 보다 낮음, 저장 안 함")
    
    # --- 5. 학습 이력 저장 ---
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # --- 6. Test 평가 (best checkpoint 로드) ---
    if splits['test']:
        print(f"\n=== 5. Test 평가 (best checkpoint) ===")
        model = AutoModelForSequenceClassification.from_pretrained(args.output).to(device)
        test_ds = SpanDataset(splits['test'], tokenizer)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        preds, labels = evaluate(model, test_loader, device, desc='Test')
        test_f1 = f1_score(labels, preds, pos_label=1)
        test_acc = accuracy_score(labels, preds)
        print(f"\nTest accuracy : {test_acc:.4f}")
        print(f"Test F1       : {test_f1:.4f}")
        print(classification_report(
            labels, preds,
            target_names=['not_in_span', 'in_span'],
            digits=4,
            zero_division=0,
        ))
    
    total_time = time.time() - start_time
    print(f"\n=== 완료 ===")
    print(f"총 학습 시간   : {total_time/60:.1f}분")
    print(f"최종 best val F1: {best_f1:.4f}")
    print(f"체크포인트     : {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1 여행 구간 이진 분류기 학습')
    parser.add_argument('--data_file', required=True,
                        help='라벨 JSON 파일 경로 (예: ./dataset/final.json)')
    parser.add_argument('--output', default='./checkpoints/stage1',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='GPU 메모리 작으면 8 또는 4로 낮추기')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_prev', type=int, default=3,
                        help='context로 쓸 직전 메시지 개수')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers (Windows에선 0)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)