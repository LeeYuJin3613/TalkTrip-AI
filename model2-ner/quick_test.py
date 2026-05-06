import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# ====================== 설정 ======================
MODEL_DIR = "ner_model_output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 모델 로드 ======================
print(f"🔧 모델 로드 중... ({MODEL_DIR})")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(DEVICE)
model.eval()
print("✅ 모델 로드 완료\n")

# ====================== 테스트 문장 ======================
test_sentences = [
    "메리 크리스마스.",
    "12월 1일 오후 6시에 쇼핑몰에서 만나자",
    "오늘 오후 3시에 카페에서 보자",
    "다음 주 일요일 오후 1시에 운동장에서 만나",
    "내일 오후 3시에 카페에서 보자",
    "이번 주 토요일 오후 1시에 부산 해운대에서 만나자",
    "6월 10일 저녁 7시에 홍대 카페에서 만나",  # 사진에 나온 문장
    "내일 오후 2시에 강남역에서 보자",
    "5월 3일 오전 10시에 서울역에서 만나자",
    "내일 오후 3시쯤에 강남역 10번 출구 앞에서 기다릴게",
    "글피 저녁 8시에 홍대입구역 근처 치킨집에서 치킨 먹자",
    "그거아냐? 내일은 사실 토요일 이란걸 말이지. 엥? 아니라고? 일요일이라고? 거짓말.",
    "글피에 시간 괜찮아?",
]

# ====================== 예측 ======================
for sentence in test_sentences:
    encoding = tokenizer(
        sentence,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offset_mapping = encoding["offset_mapping"].squeeze(0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # 확률 계산
        preds = torch.argmax(logits, dim=-1).squeeze(0)

    # 엔티티 추출 + 확률 계산
    entities = []
    current = None
    current_probs = []

    for idx, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:
            continue

        label_id = preds[idx].item()
        prob = probs[idx][label_id].item() * 100  # 퍼센트로 변환
        label = ["O", "B-DT", "I-DT", "B-LC", "I-LC", "B-TI", "I-TI"][label_id]

        if label.startswith("B-"):
            if current:
                entities.append((current["type"], current["text"].strip(), sum(current_probs) / len(current_probs)))
            current = {"type": label[2:], "text": sentence[start:end]}
            current_probs = [prob]
        elif label.startswith("I-") and current and label[2:] == current["type"]:
            current["text"] += sentence[start:end]
            current_probs.append(prob)
        else:
            if current:
                entities.append((current["type"], current["text"].strip(), sum(current_probs) / len(current_probs)))
                current = None
                current_probs = []

    if current:
        entities.append((current["type"], current["text"].strip(), sum(current_probs) / len(current_probs)))

    # 출력
    print(f"\n문장: {sentence}")
    if entities:
        for ent_type, text, confidence in entities:
            print(f"예측: [{ent_type}] {text} (확률: {confidence:.2f}%)")
    else:
        print("예측: 엔티티 없음")
    print("-" * 70)