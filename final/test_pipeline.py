"""
Stage 1+2+3 통합 추론 테스트

모든 stage를 거친 후 각 메시지에 다음 라벨이 붙음:
- in_travel_span (bool) - Stage 1
- entities (리스트, in_span만) - Stage 2
- intent (in_span만) - Stage 3

사용법:
    # 기본 테스트 케이스로 실행
    python test_pipeline.py

    # 자체 입력 파일 사용 (메시지 list 또는 {"messages":[...]} 형식)
    python test_pipeline.py --input_file my_chat.json --output_file result.json

옵션:
    --stage1, --stage2, --stage3 : 각 체크포인트 경로
    --input_file : 입력 메시지 JSON (없으면 기본 테스트)
    --output_file : 결과 저장 위치
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)


INTENTS = ['PROPOSE', 'AGREE', 'DISAGREE', 'CONFIRM', 'CANCEL', 'QUERY', 'OTHER']


class TripChatPipeline:
    """Stage 1 → 2 → 3 통합 파이프라인"""
    
    def __init__(self, stage1_dir, stage2_dir, stage3_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading from {stage1_dir}, {stage2_dir}, {stage3_dir}")
        print(f"Device: {self.device}")
        
        # Stage 1: 구간 분류
        self.span_tok = AutoTokenizer.from_pretrained(stage1_dir)
        self.span_model = AutoModelForSequenceClassification.from_pretrained(
            stage1_dir
        ).to(self.device).eval()
        
        # Stage 2: NER
        self.ner_tok = AutoTokenizer.from_pretrained(stage2_dir)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            stage2_dir
        ).to(self.device).eval()
        
        # Stage 3: Intent
        self.intent_tok = AutoTokenizer.from_pretrained(stage3_dir)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(
            stage3_dir
        ).to(self.device).eval()
        
        print("✓ 3개 모델 로딩 완료\n")

    def _format_with_context(self, text, context):
        if context:
            return ' [SEP] '.join(context) + ' [SEP] ' + text
        return text

    @torch.no_grad()
    def predict_span(self, text, context):
        """Stage 1: in_travel_span 판별"""
        full_text = self._format_with_context(text, context)
        enc = self.span_tok(
            full_text, truncation=True, padding='max_length',
            max_length=256, return_tensors='pt'
        ).to(self.device)
        logits = self.span_model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return bool(logits.argmax(-1).item()), prob

    @torch.no_grad()
    def predict_entities(self, text):
        """Stage 2: 엔티티 추출 (BIO 디코딩)"""
        enc = self.ner_tok(
            text, truncation=True, padding='max_length',
            max_length=128, return_offsets_mapping=True, return_tensors='pt'
        )
        offsets = enc.pop('offset_mapping').squeeze(0).tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.ner_model(**enc).logits
        preds = logits.argmax(-1).squeeze(0).cpu().tolist()
        id2label = self.ner_model.config.id2label
        
        # BIO 태그 → 엔티티 span 디코딩
        entities = []
        current = None
        for pred_id, (s, e) in zip(preds, offsets):
            if s == 0 and e == 0:  # padding/special
                continue
            tag = id2label[pred_id]
            if tag.startswith('B-'):
                if current:
                    entities.append(current)
                current = {'type': tag[2:], 'start': s, 'end': e}
            elif tag.startswith('I-') and current and tag[2:] == current['type']:
                current['end'] = e
            else:
                if current:
                    entities.append(current)
                current = None
        if current:
            entities.append(current)
        
        # span 인덱스로 텍스트 채우기
        for ent in entities:
            ent['text'] = text[ent['start']:ent['end']]
        return entities

    @torch.no_grad()
    def predict_intent(self, text, context):
        """Stage 3: 의도 분류"""
        full_text = self._format_with_context(text, context)
        enc = self.intent_tok(
            full_text, truncation=True, padding='max_length',
            max_length=256, return_tensors='pt'
        ).to(self.device)
        logits = self.intent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        intent_id = logits.argmax(-1).item()
        return INTENTS[intent_id], round(probs[intent_id], 4)

    def process_chat(self, messages, num_prev=3):
        """채팅 전체 처리. 직전 num_prev개 메시지를 context로 사용."""
        results = []
        for i, msg in enumerate(messages):
            text = msg if isinstance(msg, str) else msg['text']
            
            # context: 직전 N개 메시지
            context = []
            for j in range(max(0, i - num_prev), i):
                prev = messages[j]
                context.append(prev if isinstance(prev, str) else prev['text'])
            
            # Stage 1
            in_span, span_prob = self.predict_span(text, context)
            result = {
                'idx': i,
                'text': text,
                'in_travel_span': in_span,
                'span_prob': round(span_prob, 4),
            }
            
            # Stage 2 + 3 (구간 내 메시지만)
            if in_span:
                result['entities'] = self.predict_entities(text)
                intent, intent_prob = self.predict_intent(text, context)
                result['intent'] = intent
                result['intent_prob'] = intent_prob
            
            results.append(result)
        return results


def print_human_readable(results):
    """사람이 보기 좋게 출력"""
    print("\n" + "=" * 70)
    print("Stage 1+2+3 파이프라인 출력")
    print("=" * 70)
    print()
    
    for r in results:
        marker = "✓" if r['in_travel_span'] else " "
        prob = r['span_prob']
        print(f"[{marker}] (span_prob={prob:.2f}) {r['text']}")
        
        if r['in_travel_span']:
            intent = r.get('intent', 'N/A')
            iprob = r.get('intent_prob', 0)
            print(f"      → intent: {intent} ({iprob:.2f})")
            
            ents = r.get('entities', [])
            if ents:
                for e in ents:
                    print(f"      → {e['type']:10s}: {e['text']}")
            else:
                print(f"      → (엔티티 없음)")
        print()
    
    # 요약 통계
    n_total = len(results)
    n_in_span = sum(1 for r in results if r['in_travel_span'])
    n_with_ents = sum(1 for r in results 
                      if r['in_travel_span'] and r.get('entities'))
    print("=" * 70)
    print(f"요약: 메시지 {n_total}개 중 in_span {n_in_span}개, "
          f"엔티티 보유 {n_with_ents}개")
    
    # 의도 분포
    from collections import Counter
    intents = Counter(r.get('intent') for r in results if r.get('intent'))
    if intents:
        print(f"의도 분포: {dict(intents)}")


# 기본 테스트 케이스 (잡담 → 여행 협의 흐름)
DEFAULT_TEST = [
    "어제 술 너무 많이 마셨다",
    "나도 진짜 머리 아픔",
    "야 종강했는데 어디 가자",
    "맞아 어디 멀리 가고 싶다",
    "제주도 어때",
    "오 좋다",
    "2박 3일로 가자",
    "8월 첫 주말로 ㄱ",
    "성산일출봉 가자",
    "거기 새벽에 가야지",
    "또 일출봉이냐",
    "흑돼지도 먹어야 함",
    "오케이",
    "비행기는 제주항공",
    "근데 나 늦으면 그냥 가",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1', default='./checkpoints/stage1')
    parser.add_argument('--stage2', default='./checkpoints/stage2')
    parser.add_argument('--stage3', default='./checkpoints/stage3')
    parser.add_argument('--input_file', default=None,
                        help='입력 메시지 JSON (list 또는 {"messages":[...]})')
    parser.add_argument('--output_file', default=None,
                        help='결과 JSON 저장 위치 (선택)')
    args = parser.parse_args()
    
    pipeline = TripChatPipeline(args.stage1, args.stage2, args.stage3)
    
    if args.input_file:
        with open(args.input_file, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'messages' in data:
            messages = [m['text'] if isinstance(m, dict) else m 
                        for m in data['messages']]
        elif isinstance(data, list):
            messages = [m['text'] if isinstance(m, dict) else m for m in data]
        else:
            raise ValueError("입력 형식 미지원")
        print(f"입력 파일에서 {len(messages)}개 메시지 로드")
    else:
        messages = DEFAULT_TEST
        print(f"기본 테스트 케이스 사용 ({len(messages)}개 메시지)")
    
    results = pipeline.process_chat(messages)
    print_human_readable(results)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON 결과 저장: {args.output_file}")