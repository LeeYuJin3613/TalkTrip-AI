"""
Stage 2: NER - 9가지 엔티티 인식 (BIO tagging)

엔티티 타입 (9개):
    LOC, DATE, TIME, DURATION, TRANSPORT, LODGING, ACTIVITY, FOOD, COST

라벨 데이터의 entities는 [{type, text}] 형식 (character offset 없음).
text.find()로 출현 위치를 찾아 토큰 단위 BIO 라벨로 변환.

핵심 디자인:
- 입력: 단일 메시지 (context 불필요 - 엔티티는 메시지 안에서만 추출)
- 모델: KLUE-RoBERTa + token classification head
- 출력: 토큰별 BIO 태그 (O + 9*2 = 19 라벨)
- 평가: seqeval entity-level F1 (token-level 아닌 strict matching)

사용법:
    pip install torch transformers scikit-learn seqeval tqdm
    python stage2.py --data_file ./dataset/final.json
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score


MODEL_NAME = 'klue/roberta-base'
MAX_LENGTH = 128

ENTITY_TYPES = ['LOC', 'DATE', 'TIME', 'DURATION', 'TRANSPORT',
                'LODGING', 'ACTIVITY', 'FOOD', 'COST']
LABELS = ['O'] + [f'{p}-{e}' for e in ENTITY_TYPES for p in ['B', 'I']]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


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
    """엔티티 통계 포함"""
    for name in ['train', 'val', 'test']:
        msgs = splits[name]
        n_total = len(msgs)
        n_with_ent = sum(1 for m in msgs if m.get('entities'))
        n_entities = sum(len(m.get('entities', [])) for m in msgs)
        n_chats = len(set(m['source'] for m in msgs))
        print(f"  [{name:5s}] 채팅 {n_chats}개, 메시지 {n_total}개, "
              f"엔티티 보유 메시지 {n_with_ent}개, 엔티티 총 {n_entities}개")


# ============================================================
# BIO 변환 - 핵심 로직
# ============================================================

def find_entity_span(text, entity_text, used_spans):
    """
    text 안에서 entity_text의 첫 출현 위치 반환.
    이미 다른 엔티티가 차지한 span은 피해서 다음 출현을 찾음.
    """
    start = 0
    while True:
        idx = text.find(entity_text, start)
        if idx == -1:
            return None, None
        end = idx + len(entity_text)
        # 다른 엔티티와 겹치지 않으면 사용
        overlap = any(not (end <= s or idx >= e) for s, e in used_spans)
        if not overlap:
            return idx, end
        start = idx + 1


def text_to_bio(text, entities, tokenizer, max_length=MAX_LENGTH):
    """문자열 + 엔티티 리스트 → 토큰 BIO 라벨"""
    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors='pt',
    )
    offsets = enc['offset_mapping'].squeeze(0).tolist()
    labels = ['O'] * len(offsets)
    used_spans = []
    
    # 긴 엔티티 먼저 처리 (예: "춘천 벨라 레지던스 호텔"이 "춘천"보다 우선)
    sorted_entities = sorted(entities, key=lambda e: -len(e['text']))
    
    for ent in sorted_entities:
        ent_text, ent_type = ent['text'], ent['type']
        if ent_type not in ENTITY_TYPES:
            continue
        start, end = find_entity_span(text, ent_text, used_spans)
        if start is None:
            continue
        used_spans.append((start, end))
        
        # 문자 offset → 토큰 인덱스 매핑 + B-/I- 태깅
        first = True
        for i, (ts, te) in enumerate(offsets):
            if ts == 0 and te == 0:  # padding/special token
                continue
            if ts >= start and te <= end:
                labels[i] = f'B-{ent_type}' if first else f'I-{ent_type}'
                first = False
    
    label_ids = [LABEL2ID[l] for l in labels]
    # padding/special token은 -100으로 마스킹 (loss 계산 제외)
    for i, (ts, te) in enumerate(offsets):
        if ts == 0 and te == 0:
            label_ids[i] = -100
    
    return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0), torch.tensor(label_ids)


# ============================================================
# Dataset
# ============================================================

class NERDataset(Dataset):
    """
    학습 대상: 구간 내 메시지 + 엔티티 보유 메시지.
    잡담 메시지는 대부분 엔티티 없어서 모델이 "전부 O" 학습하는 노이즈가 됨.
    """
    def __init__(self, messages, tokenizer, max_length=MAX_LENGTH):
        self.messages = [m for m in messages 
                         if m.get('in_travel_span') or m.get('entities')]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        m = self.messages[idx]
        input_ids, attention_mask, labels = text_to_bio(
            m['text'], m.get('entities', []), self.tokenizer, self.max_length
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# 평가
# ============================================================

def evaluate(model, loader, device, desc='Eval'):
    """seqeval entity-level F1 계산"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            for p_seq, l_seq in zip(preds, labels):
                p_tags, l_tags = [], []
                for p, l in zip(p_seq, l_seq):
                    if l == -100:
                        continue
                    p_tags.append(ID2LABEL[int(p)])
                    l_tags.append(ID2LABEL[int(l)])
                all_preds.append(p_tags)
                all_labels.append(l_tags)
    return all_preds, all_labels


# ============================================================
# 학습
# ============================================================

def train(args):
    print("\n=== 1. 데이터 로드 ===")
    messages = load_data(args.data_file)
    splits = split_by_chat(messages, seed=args.seed)
    
    print("\n=== 2. 데이터 분할 ===")
    print_data_stats(splits)
    
    print(f"\n=== 3. 모델 로딩 ({MODEL_NAME}) ===")
    print("  (첫 실행 시 ~440MB 다운로드)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Device      : {device}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Labels      : {len(LABELS)} (O + {len(ENTITY_TYPES)}*B/I)")
    
    train_ds = NERDataset(splits['train'], tokenizer)
    val_ds = NERDataset(splits['val'], tokenizer)
    print(f"  NER train   : {len(train_ds)} (구간내+엔티티 보유 필터링 후)")
    print(f"  NER val     : {len(val_ds)}")
    
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
        preds, labels = evaluate(model, val_loader, device, desc='Val')
        f1 = f1_score(labels, preds)
        elapsed = time.time() - start_time
        
        print(f"\n[Epoch {epoch+1}] loss={avg_loss:.4f}  val_f1={f1:.4f}  ({elapsed:.0f}s)")
        print(classification_report(labels, preds, digits=3, zero_division=0))
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_f1': f1,
        })
        
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ✓ Best F1 갱신 → {args.output}/ 에 저장")
        else:
            print(f"  · 이전 best F1 ({best_f1:.4f}) 보다 낮음, 저장 안 함")
    
    with open(os.path.join(args.output, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    if splits['test']:
        print(f"\n=== 5. Test 평가 (best checkpoint) ===")
        model = AutoModelForTokenClassification.from_pretrained(args.output).to(device)
        test_ds = NERDataset(splits['test'], tokenizer)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
        preds, labels = evaluate(model, test_loader, device, desc='Test')
        test_f1 = f1_score(labels, preds)
        print(f"\nTest F1 (entity-level): {test_f1:.4f}")
        print(classification_report(labels, preds, digits=3, zero_division=0))
    
    total_time = time.time() - start_time
    print(f"\n=== 완료 ===")
    print(f"총 학습 시간   : {total_time/60:.1f}분")
    print(f"최종 best val F1: {best_f1:.4f}")
    print(f"체크포인트     : {args.output}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 NER 학습')
    parser.add_argument('--data_file', required=True,
                        help='라벨 JSON 파일 경로 (예: ./dataset/final.json)')
    parser.add_argument('--output', default='./checkpoints/stage2',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--epochs', type=int, default=10,
                        help='NER은 stage1보다 더 많은 epoch 필요')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='GPU 메모리 작으면 8 또는 4로')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='NER은 약간 높은 lr 권장')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42,
                        help='stage1과 같은 seed 권장 (같은 분할 보장)')
    args = parser.parse_args()
    train(args)