"""
=============================================================================
Korean BERT NER 추론(테스트) 스크립트 (v2 — 컨텍스트 윈도우 지원)
- 학습된 모델로 문장에서 날짜(DT), 장소(LC), 시간(TI) 추출
- 앞뒤 맥락 문장을 포함하여 추론 정확도 향상
- 입력: 문장(+맥락) → 출력: {"날짜": [...], "장소": [...], "시간": [...]}
=============================================================================

[사용법]
  python test_ner.py
  python test_ner.py --model_dir ner_model_output
  python test_ner.py --sentence "내일 오후 3시에 서울역에서 만나자"

  # 맥락 포함 추론 (| 로 문장 구분, 마지막 문장이 타겟)
  python test_ner.py --context "주말에 뭐 할까?|내일 오후 3시에 서울역에서 만나자"
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL_LIST = ["O", "B-DT", "I-DT", "B-LC", "I-LC", "B-TI", "I-TI"]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
MAX_LEN = 256
CONTEXT_WINDOW = 2  # 학습 시 사용한 값과 동일하게

# 엔티티 타입 → 한글 표시
TYPE_NAME = {"DT": "날짜", "LC": "장소", "TI": "시간"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  추론 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NERPredictor:
    def __init__(self, model_dir, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # ←←← 절대경로로 강제 변환 + 디버그 출력 추가
        import os
        model_dir = os.path.abspath(model_dir)
        print(f"🔧 모델 로드 시도 경로: {model_dir}")
        print(f"   → model.safetensors 존재 여부: {os.path.exists(os.path.join(model_dir, 'model.safetensors'))}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()
        print("✅ 모델 로드 완료 (로컬 모델 사용)")

    def predict(self, sentence: str,
                prev_sentences: list = None,
                next_sentences: list = None) -> dict:
        """
        문장 → 엔티티 추출 (선택적으로 앞뒤 맥락 문장 포함)

        Args:
            sentence:        타겟 문장
            prev_sentences:  앞쪽 맥락 문장 리스트 (먼 것부터 순서대로)
            next_sentences:  뒷쪽 맥락 문장 리스트 (가까운 것부터 순서대로)

        Returns:
            {
                "sentence": "원본 문장",
                "entities": [...],
                "날짜": [...], "장소": [...], "시간": [...]
            }
        """
        prev = prev_sentences or []
        nxt = next_sentences or []

        # 컨텍스트 포함한 전체 텍스트 구성 & 타겟 위치 추적
        parts = prev + [sentence] + nxt
        target_idx_in_parts = len(prev)

        full_text = ""
        target_start = -1
        target_end = -1

        for j, part in enumerate(parts):
            if j > 0:
                full_text += " "
            if j == target_idx_in_parts:
                target_start = len(full_text)
            full_text += part
            if j == target_idx_in_parts:
                target_end = len(full_text)

        # BERT 토큰화
        encoding = self.tokenizer(
            full_text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1).squeeze(0)
            max_probs = probs.squeeze(0).max(dim=-1).values

        # subword 예측 → 타겟 문장 문자 단위로 매핑
        char_labels = ["O"] * len(sentence)
        char_probs = [0.0] * len(sentence)

        for idx, (start, end) in enumerate(offset_mapping):
            s, e = start.item(), end.item()
            if s == 0 and e == 0:
                continue
            # 타겟 범위 내 토큰만 처리
            if s >= target_start and e <= target_end:
                label = ID2LABEL[preds[idx].item()]
                prob = max_probs[idx].item()
                for ci in range(s - target_start, min(e - target_start, len(sentence))):
                    if char_labels[ci] == "O" or prob > char_probs[ci]:
                        char_labels[ci] = label
                        char_probs[ci] = prob

        # BIO 태그에서 엔티티 추출
        entities = self._extract_entities(sentence, char_labels)

        return {
            "sentence": sentence,
            "entities": entities,
            "날짜": [e["text"] for e in entities if e["type"] == "DT"],
            "장소": [e["text"] for e in entities if e["type"] == "LC"],
            "시간": [e["text"] for e in entities if e["type"] == "TI"],
        }

    def predict_in_sequence(self, sentences: list, window: int = 2) -> list:
        """
        연속된 문장 리스트에서 각 문장마다 앞뒤 window 문장을
        맥락으로 포함하여 추론

        Args:
            sentences: 연속 문장 리스트
            window:    앞뒤 맥락 윈도우 크기

        Returns:
            각 문장의 추론 결과 리스트
        """
        results = []
        for i, sent in enumerate(sentences):
            prev = sentences[max(0, i - window):i]
            nxt = sentences[i + 1:i + 1 + window]
            result = self.predict(sent, prev_sentences=prev, next_sentences=nxt)
            results.append(result)
        return results

    @staticmethod
    def _extract_entities(text, char_labels):
        """BIO 라벨 시퀀스 → 엔티티 리스트"""
        entities = []
        current = None

        for i, (ch, label) in enumerate(zip(text, char_labels)):
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"type": label[2:], "text": ch}
            elif label.startswith("I-") and current:
                if label[2:] == current["type"]:
                    current["text"] += ch
                else:
                    entities.append(current)
                    current = None
            else:
                if current:
                    entities.append(current)
                    current = None

        if current:
            entities.append(current)

        # 공백 정리 & 한글 라벨
        clean = []
        for ent in entities:
            ent["text"] = ent["text"].strip()
            if ent["text"]:
                ent["label"] = TYPE_NAME.get(ent["type"], ent["type"])
                clean.append(ent)
        return clean


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  테스트 데이터셋 정량 평가 (컨텍스트 윈도우 적용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate_on_testset(predictor, data_path, window=2, seed=42):
    """
    학습 시와 동일한 분할 기준으로 테스트셋 추출 후
    컨텍스트 윈도우를 적용하여 엔티티 단위 정확도 평가
    """
    from sklearn.model_selection import train_test_split
    from seqeval.metrics import classification_report
    from seqeval.scheme import IOB2

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 동일한 분할
    train_val, test_data = train_test_split(data, test_size=0.15, random_state=seed)

    print(f"\n📊 테스트셋 평가 ({len(test_data)}개 샘플, 컨텍스트 윈도우={window})")
    print("─" * 55)

    all_true, all_pred = [], []
    correct_entities = 0
    total_true_entities = 0
    total_pred_entities = 0

    for i, item in enumerate(test_data):
        text = "".join(item["tokens"])
        true_labels = item["labels"]

        # 앞뒤 맥락 문장 구성 (테스트셋 내에서)
        prev_sents = []
        for j in range(max(0, i - window), i):
            prev_sents.append("".join(test_data[j]["tokens"]))

        next_sents = []
        for j in range(i + 1, min(len(test_data), i + 1 + window)):
            next_sents.append("".join(test_data[j]["tokens"]))

        result = predictor.predict(
            text,
            prev_sentences=prev_sents,
            next_sentences=next_sents,
        )

        # 정답 엔티티 추출
        true_entities = []
        current = None
        for t, l in zip(item["tokens"], true_labels):
            if l.startswith("B-"):
                if current:
                    true_entities.append(current)
                current = {"type": l[2:], "text": t}
            elif l.startswith("I-") and current:
                current["text"] += t
            else:
                if current:
                    true_entities.append(current)
                    current = None
        if current:
            true_entities.append(current)
        true_entities = [e for e in true_entities if e["text"].strip()]

        # 예측 엔티티
        pred_entities = result["entities"]

        # 매칭 (type + text 정확 일치)
        true_set = {(e["type"], e["text"].strip()) for e in true_entities}
        pred_set = {(e["type"], e["text"].strip()) for e in pred_entities}

        correct_entities += len(true_set & pred_set)
        total_true_entities += len(true_set)
        total_pred_entities += len(pred_set)

        # seqeval용 시퀀스
        all_true.append(true_labels)
        pred_char_labels = ["O"] * len(item["tokens"])
        for ent in pred_entities:
            start_idx = text.find(ent["text"])
            if start_idx >= 0:
                pred_char_labels[start_idx] = f"B-{ent['type']}"
                for k in range(1, len(ent["text"])):
                    if start_idx + k < len(pred_char_labels):
                        pred_char_labels[start_idx + k] = f"I-{ent['type']}"
        all_pred.append(pred_char_labels)

    # 엔티티 단위 정확도
    entity_precision = correct_entities / max(total_pred_entities, 1)
    entity_recall = correct_entities / max(total_true_entities, 1)
    entity_f1 = (
        2 * entity_precision * entity_recall / max(entity_precision + entity_recall, 1e-8)
    )

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  엔티티 단위 (exact match) 평가 결과      │")
    print(f"  ├──────────────────────────────────────────┤")
    print(f"  │  Precision : {entity_precision:.4f}                       │")
    print(f"  │  Recall    : {entity_recall:.4f}                       │")
    print(f"  │  F1-Score  : {entity_f1:.4f}                       │")
    print(f"  │                                          │")
    print(f"  │  정답 엔티티  : {total_true_entities:>6}개                  │")
    print(f"  │  예측 엔티티  : {total_pred_entities:>6}개                  │")
    print(f"  │  정확 매칭    : {correct_entities:>6}개                  │")
    print(f"  └──────────────────────────────────────────┘")

    # seqeval 리포트
    print("\n  === seqeval 토큰 단위 상세 리포트 ===\n")
    report = classification_report(all_true, all_pred, mode="strict", scheme=IOB2, digits=4)
    print(report)

    return {
        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "entity_f1": round(entity_f1, 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  실제 문장 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_demo(predictor):
    """
    실제 대화 시나리오 데모
    - 단일 문장 추론 & 연속 대화 맥락 추론 비교
    """
    # 1) 단일 문장 테스트
    single_sentences = [
        "내일 오후 3시에 서울역에서 만나자",
        "12월 25일 크리스마스에 부산 해운대로 여행 갈까?",
        "2025년 1월 15일에 제주도 여행 예약했어",
        "오늘 저녁 7시에 홍대입구역 근처에서 저녁 먹자",
    ]

    print("\n" + "=" * 65)
    print("  🧪 단일 문장 테스트")
    print("=" * 65)

    for sent in single_sentences:
        result = predictor.predict(sent)
        print(f"\n  문장: {sent}")
        print(f"  날짜: {result['날짜'] or '없음'}")
        print(f"  장소: {result['장소'] or '없음'}")
        print(f"  시간: {result['시간'] or '없음'}")
        print("  " + "─" * 50)

    # 2) 연속 대화 맥락 테스트
    conversation = [
        "이번 주말에 어디 놀러 갈까?",
        "제주도 어때? 날씨도 좋다던데",
        "좋아 그러면 토요일 아침 8시 비행기 타자",
        "김포공항에서 출발하면 되겠다",
        "숙소는 서귀포 쪽으로 잡을까?",
    ]

    print("\n" + "=" * 65)
    print("  🧪 연속 대화 맥락 테스트 (윈도우=2)")
    print("=" * 65)

    results = predictor.predict_in_sequence(conversation, window=CONTEXT_WINDOW)
    for sent, result in zip(conversation, results):
        print(f"\n  문장: {sent}")
        print(f"  날짜: {result['날짜'] or '없음'}")
        print(f"  장소: {result['장소'] or '없음'}")
        print(f"  시간: {result['시간'] or '없음'}")
        if result["entities"]:
            for ent in result["entities"]:
                print(f"    → [{ent['label']}] {ent['text']}")
        print("  " + "─" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  대화형 모드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def interactive_mode(predictor):
    """
    대화형 NER 추론
    - 이전 입력을 맥락으로 자동 유지 (최대 window개)
    """
    print("\n" + "=" * 65)
    print("  💬 대화형 NER 추론 모드 (맥락 자동 유지)")
    print("  문장을 입력하면 이전 대화를 맥락으로 활용합니다.")
    print("  'clear' → 맥락 초기화")
    print("  'quit' 또는 'q' → 종료")
    print("=" * 65)

    history = []

    while True:
        sentence = input("\n입력 > ").strip()
        if sentence.lower() in ("quit", "q", "exit"):
            print("종료합니다.")
            break
        if sentence.lower() == "clear":
            history.clear()
            print("  🔄 맥락 초기화 완료")
            continue
        if not sentence:
            continue

        # 이전 대화를 맥락으로 사용
        prev = history[-CONTEXT_WINDOW:] if history else []
        result = predictor.predict(sentence, prev_sentences=prev)

        print(f"  날짜: {result['날짜'] or '없음'}")
        print(f"  장소: {result['장소'] or '없음'}")
        print(f"  시간: {result['시간'] or '없음'}")

        if prev:
            print(f"  (맥락: 이전 {len(prev)}문장 참조)")

        history.append(sentence)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Korean BERT NER 추론 (컨텍스트 윈도우)")
    parser.add_argument("--model_dir", type=str, default="ner_model_output", help="학습된 모델 디렉토리")
    parser.add_argument("--data_path", type=str, default="bert_ner.json", help="평가용 원본 데이터 경로")
    parser.add_argument("--sentence", type=str, default=None, help="단일 문장 추론")
    parser.add_argument("--context", type=str, default=None,
                        help="맥락 포함 추론 ('|'로 구분, 마지막이 타겟). 예: '앞문장|타겟문장'")
    parser.add_argument("--window", type=int, default=CONTEXT_WINDOW, help="컨텍스트 윈도우 크기")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    parser.add_argument("--eval_only", action="store_true", help="테스트셋 정량 평가만 수행")
    parser.add_argument("--demo_only", action="store_true", help="데모 문장만 실행")
    args = parser.parse_args()

    predictor = NERPredictor(args.model_dir)

    if args.context:
        # 맥락 포함 추론 (| 구분)
        parts = [p.strip() for p in args.context.split("|")]
        target = parts[-1]
        prev = parts[:-1]
        result = predictor.predict(target, prev_sentences=prev)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.sentence:
        # 단일 문장 추론 (맥락 없음)
        result = predictor.predict(args.sentence)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.eval_only:
        evaluate_on_testset(predictor, args.data_path, window=args.window)

    elif args.demo_only:
        run_demo(predictor)

    elif args.interactive:
        interactive_mode(predictor)

    else:
        # 기본: 정량 평가 + 데모 + 대화형
        evaluate_on_testset(predictor, args.data_path, window=args.window)
        run_demo(predictor)
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
