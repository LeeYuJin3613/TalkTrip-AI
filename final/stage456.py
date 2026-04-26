"""
Stage 4 + 5 + 6: 정규화 + 이벤트 조립 + 일자별 일정표 생성 (개선판)

주요 개선:
1. destination 결정: 도시급 LOC 우선 (sub-location은 후보에서 제외)
2. AGREE/PROPOSE 메시지의 entity 모두 활용 (메타 정보 빠뜨리지 않음)
3. "둘째 날", "셋째날" 같은 명시적 day 키워드 만나면 그 이후 메시지를 해당 day로
4. duration이 메시지 intent와 무관하게 추출됨

사용법:
    python stage456.py --pred result.json --output summary.json --verbose
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta


# ============================================================
# 상수
# ============================================================

DEFAULT_BASE_DATE = datetime(2024, 8, 1)
WEEKDAYS = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}

PERIOD_TO_TIME = {
    '새벽': '05:00', '아침': '08:00', '오전': '10:00', '점심': '12:00',
    '오후': '14:00', '저녁': '18:00', '밤': '21:00', '야간': '22:00',
}

# 명시적 day 키워드 (텍스트에서 검색)
DAY_PATTERNS = [
    (r'첫째?\s*날', 1),
    (r'둘째\s*날', 2),
    (r'셋째\s*날', 3),
    (r'넷째\s*날', 4),
    (r'다섯째\s*날', 5),
    (r'마지막\s*날|마지막날', -1),  # -1은 마지막을 의미
    (r'다음\s*날', 'NEXT'),  # 직전 day + 1
]

CATEGORY_MAP = {
    'LOC': 'PLACE',
    'LODGING': 'LODGING',
    'FOOD': 'FOOD',
    'ACTIVITY': 'ACTIVITY',
    'TRANSPORT': 'TRANSPORT',
}

META_TYPES = {'DURATION', 'DATE', 'COST'}

# 도시급 LOC 키워드 (destination 후보로 가산점)
CITY_KEYWORDS = ['시', '도', '제주', '서울', '부산', '강릉', '경주', '여수', '속초',
                  '춘천', '포항', '공주', '부여', '제천', '가평', '양양', '용인',
                  '대전', '대구', '광주', '인천', '울산', '수원']

# 명백한 sub-location (destination에서 제외)
SUB_LOC_KEYWORDS = ['역', '터미널', '공항', '근처', '근방', '시장', '카페',
                     '카페거리', '거리', '해변', '해수욕장', '해안', '바다',
                     '호수', '산', '봉', '대교', '다리', '광장', '공원']


# 출발지 표현 패턴 (여행 시작점)
# {loc} 다음에 '쪽', 공백 등이 와도 매칭
DEPARTURE_PATTERNS = [
    r'{loc}\s*(쪽\s*)?에서\s*출발',     # "청주에서 출발", "청주 쪽에서 출발"
    r'{loc}\s*(쪽\s*)?에서\s*만나',     # "터미널에서 만나", "터미널 쪽에서 만나"
]

# 귀환지/경유지 표현 패턴
RETURN_PATTERNS = [
    r'{loc}\s*(쪽\s*)?로\s*가는\s*길',
    r'{loc}\s*(쪽\s*)?으?로\s*가는\s*길',
    r'{loc}\s*(쪽\s*)?로\s*돌아',
    r'{loc}\s*(쪽\s*)?으?로\s*돌아',
    r'{loc}\s*(쪽\s*)?로\s*복귀',
    r'{loc}\s*(쪽\s*)?으?로\s*복귀',
    r'{loc}\s*올라가',
]


def detect_loc_role(text, loc):
    """
    LOC가 메시지에서 어떤 역할로 쓰였는지 판별.
    Returns: 'departure' | 'return' | 'event' (일반 일정 항목)
    """
    import re as _re
    loc_escaped = _re.escape(loc)
    
    for pattern in DEPARTURE_PATTERNS:
        if _re.search(pattern.format(loc=loc_escaped), text):
            return 'departure'
    
    for pattern in RETURN_PATTERNS:
        if _re.search(pattern.format(loc=loc_escaped), text):
            return 'return'
    
    return 'event'


# 숙소 위치 의논 맥락 키워드 (직전 메시지에 있으면 다음 LOC도 숙소 컨텍스트)
LODGING_CONTEXT_KEYWORDS = ['숙소', '묵을', '잡을', '잡자', '숙박']

# 숙소 위치 추천 패턴 (메시지가 이렇게 끝나면 진짜 활동지가 아닌 숙소 후보 제안)
LODGING_LOC_PATTERNS = [
    r'{loc}\s*쪽',                  # "안목해변 쪽이"
    r'{loc}\s*근처',                # "강릉역 근처"
]


def is_lodging_context_loc(msg_idx, in_span_msgs, loc_text, lookback=3):
    """
    LOC가 숙소 위치 의논 맥락에서 나온 건지 판별.
    
    조건 (둘 다 만족해야 함):
    1. 메시지 텍스트가 "쪽이/근처가" 같은 추천 패턴
    2. 직전 lookback개 메시지에 LODGING 관련 단서 있음
    """
    import re as _re
    
    if msg_idx < 0 or msg_idx >= len(in_span_msgs):
        return False
    
    text = in_span_msgs[msg_idx]['text']
    loc_escaped = _re.escape(loc_text)
    
    # 조건 1: 추천 패턴
    has_recommend_pattern = any(
        _re.search(p.format(loc=loc_escaped), text)
        for p in LODGING_LOC_PATTERNS
    )
    if not has_recommend_pattern:
        return False
    
    # 조건 2: 직전 lookback개 메시지에 LODGING 컨텍스트
    start = max(0, msg_idx - lookback)
    for j in range(start, msg_idx):
        prev_text = in_span_msgs[j]['text']
        prev_ents = in_span_msgs[j].get('entities', [])
        
        # 키워드 매칭
        if any(kw in prev_text for kw in LODGING_CONTEXT_KEYWORDS):
            return True
        # LODGING entity 매칭
        if any(e['type'] == 'LODGING' for e in prev_ents):
            return True
    
    return False


# ============================================================
# Stage 4: 정규화
# ============================================================

def normalize_time(text):
    text = text.strip()
    
    m = re.search(r'오후\s*(\d{1,2})시', text)
    if m:
        h = int(m.group(1))
        if h != 12:
            h += 12
        return f"{h:02d}:00"
    
    m = re.search(r'오전\s*(\d{1,2})시', text)
    if m:
        h = int(m.group(1))
        if h == 12:
            h = 0
        return f"{h:02d}:00"
    
    m = re.search(r'(\d{1,2})시\s*반', text)
    if m:
        return f"{int(m.group(1)):02d}:30"
    
    m = re.search(r'(\d{1,2})시', text)
    if m:
        return f"{int(m.group(1)):02d}:00"
    
    m = re.search(r'(\d{1,2})\s*[~-]\s*\d{1,2}시', text)
    if m:
        return f"{int(m.group(1)):02d}:00"
    
    for k, v in PERIOD_TO_TIME.items():
        if k in text:
            return v
    
    return None


def normalize_date(text, base_date=DEFAULT_BASE_DATE):
    text = text.strip()
    
    m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            d = datetime(base_date.year, month, day)
            if d < base_date - timedelta(days=30):
                d = datetime(base_date.year + 1, month, day)
            return d.strftime('%Y-%m-%d')
        except ValueError:
            return text
    
    m = re.search(r'(이번|다음)\s*주\s*([월화수목금토일])', text)
    if m:
        which = m.group(1)
        weekday = WEEKDAYS[m.group(2)]
        days_until = (weekday - base_date.weekday()) % 7
        if which == '다음':
            days_until += 7
        elif days_until == 0:
            days_until = 7
        target = base_date + timedelta(days=days_until)
        return target.strftime('%Y-%m-%d')
    
    return text


def normalize_duration(text):
    text = text.strip()
    
    m = re.search(r'(\d+)박\s*(\d+)일', text)
    if m:
        return {'nights': int(m.group(1)), 'days': int(m.group(2)),
                'display': f"{m.group(1)}박 {m.group(2)}일"}
    
    if '당일' in text:
        return {'nights': 0, 'days': 1, 'display': '당일치기'}
    
    return {'display': text}


# ============================================================
# Stage 5 보조: day 추정
# ============================================================

def detect_day_keyword(text, current_day):
    """메시지 텍스트에서 명시적 day 키워드 찾기"""
    for pattern, day_val in DAY_PATTERNS:
        if re.search(pattern, text):
            if day_val == 'NEXT':
                return current_day + 1
            return day_val
    return None


def detect_message_time(text, entities):
    """메시지의 시간 추정"""
    for ent in entities:
        if ent['type'] == 'TIME':
            t = normalize_time(ent['text'])
            if t:
                return t
    
    for k, v in PERIOD_TO_TIME.items():
        if k in text:
            return v
    
    return None


def time_to_minutes(time_str):
    if not time_str:
        return 9999
    try:
        h, m = time_str.split(':')
        return int(h) * 60 + int(m)
    except:
        return 9999


def is_sub_location(text):
    """sub-location(역/시장/카페 등) 여부"""
    return any(kw in text for kw in SUB_LOC_KEYWORDS)


def is_city_location(text):
    """도시급 LOC 여부"""
    if is_sub_location(text):
        return False
    return any(kw in text for kw in CITY_KEYWORDS)


def is_address(text):
    """주소 형식 (도/시/구/로 등 포함, 숫자 포함)"""
    addr_keywords = ['도 ', '시 ', '구 ', '로 ', '읍', '면', '동']
    has_addr_kw = any(kw in text for kw in addr_keywords)
    has_num = bool(re.search(r'\d', text))
    return has_addr_kw and has_num


def extract_departure_return(messages):
    """
    출발 정보(departure)와 귀환 경유지(return_via) 추출.
    """
    departure_loc = None
    departure_time = None
    return_locs = []
    
    in_span_msgs = [m for m in messages if m.get('in_travel_span')]
    departure_msg_indices = []  # 출발지 패턴이 매칭된 모든 메시지 index
    
    # 1차: 출발지/귀환지 식별
    for i, msg in enumerate(in_span_msgs):
        text = msg['text']
        ents = msg.get('entities', [])
        
        for ent in ents:
            if ent['type'] != 'LOC':
                continue
            role = detect_loc_role(text, ent['text'])
            
            if role == 'departure':
                if departure_loc is None:
                    departure_loc = ent['text']
                departure_msg_indices.append(i)
            
            elif role == 'return':
                if ent['text'] not in return_locs:
                    return_locs.append(ent['text'])
    
    # 2차: departure_loc과 같이 등장하는 메시지 중 TIME 있는 것 우선
    # (예: idx 82 "서울역에서 만나자"에는 시간 없지만, idx 86 "9시 반쯤 서울역에서 만나자"엔 있음)
    if departure_loc:
        # (a) 출발지 LOC이 다시 나오는 메시지에서 TIME 찾기
        for i, msg in enumerate(in_span_msgs):
            if departure_loc in msg['text']:
                for ent in msg.get('entities', []):
                    if ent['type'] == 'TIME':
                        t = normalize_time(ent['text'])
                        if t:
                            # 같은 메시지에 출발지+시간 있으면 가장 우선
                            departure_time = t
                            break
            if departure_time:
                break
        
        # (b) 못 찾았으면 출발지 메시지 인접(±5)에서 검색
        if not departure_time and departure_msg_indices:
            first_dep_idx = departure_msg_indices[0]
            search_start = max(0, first_dep_idx - 5)
            search_end = min(len(in_span_msgs), first_dep_idx + 6)
            for j in range(search_start, search_end):
                for ent in in_span_msgs[j].get('entities', []):
                    if ent['type'] == 'TIME':
                        t = normalize_time(ent['text'])
                        if t:
                            departure_time = t
                            break
                if departure_time:
                    break
    
    departure = None
    if departure_loc:
        departure = {'location': departure_loc, 'time': departure_time}
    
    return departure, return_locs


# ============================================================
# Stage 5: 이벤트 조립 (개선)
# ============================================================

def assemble_events(messages, lookback=5):
    """
    PROPOSE/AGREE/CONFIRM 모두 추적.
    
    개선점:
    1. AGREE/CONFIRM 만나면 직전 "연속 PROPOSE 블록" 전체를 confirm
       (한 묶음 + AGREE 하나 패턴 지원)
    2. PROPOSE 텍스트에 동의 표현(ㄱㄱ/콜/ㅇㅋ)이 포함되면 직전 PROPOSE도 같이 confirm
       (구체화 + 동의 패턴 지원)
    3. 숙소 위치 의논 맥락의 LOC는 entities에서 제거 (예: "안목해변 쪽이 좋지 않냐")
    4. 일반 → 구체화 흡수: 직전 PROPOSE의 entity가 현재 entity의 부분 문자열이면
       일반 버전을 'absorbed'로 표시 (일정에서 제외)
       (예: "초당순두부" → "초당할머니순두부 ㄱㄱ" 시 초당순두부 흡수)
    """
    proposals = []
    current_day = 1
    
    AGREE_TOKENS = ['ㄱㄱ', 'ㅇㅋ', '콜', 'ㄱㄱㄱ', 'ㄱㄱㄱㄱ']
    
    in_span_msgs = [m for m in messages if m.get('in_travel_span')]
    msg_to_span_idx = {}
    for span_idx, m in enumerate(in_span_msgs):
        msg_to_span_idx[id(m)] = span_idx
    
    def confirm_block(end_idx, response_text):
        block_props = []
        last_idx = end_idx
        for p in reversed(proposals):
            if p['status'] != 'pending':
                continue
            if (end_idx - p['idx']) > lookback:
                break
            if last_idx - p['idx'] > 3:
                break
            block_props.append(p)
            last_idx = p['idx']
        for p in block_props:
            p['status'] = 'confirmed'
            p['response_text'] = response_text
    
    def is_subsumed_by(prev_text, cur_texts):
        """
        prev_text가 cur_texts 중 하나에 의해 흡수되는지 판별.
        
        흡수 조건:
        1. 정확히 같음 (중복): "닭강정" == "닭강정" → 흡수
        2. 부분 문자열 (포함): "닭강정" ⊂ "만석닭강정" → 흡수
        3. 공통 접두/접미사가 길면: "초당순두부"와 "초당할머니순두부"는 
           접두 "초당"(2자)+접미 "순두부"(3자) 공유, prev 길이의 80%+ 공유 → 흡수
        """
        for cur in cur_texts:
            if prev_text == cur:
                return True  # 중복
            if prev_text != cur and prev_text in cur:
                return True  # 포함
            
            # 공통 접두/접미사 길이 계산
            prefix_len = 0
            for a, b in zip(prev_text, cur):
                if a == b:
                    prefix_len += 1
                else:
                    break
            
            suffix_len = 0
            for a, b in zip(reversed(prev_text), reversed(cur)):
                if a == b:
                    suffix_len += 1
                else:
                    break
            
            # 양쪽 끝 글자가 모두 일치하면서 합쳐서 prev의 80% 이상이면 흡수
            shared = prefix_len + suffix_len
            if (prefix_len >= 1 and suffix_len >= 1 
                    and shared >= max(2, len(prev_text) * 0.8)):
                return True
        
        return False
    
    def absorb_general_by_specific(current_proposal):
        """
        현재 PROPOSE에 동의 토큰이 있고, entity가 직전 PROPOSE entity의 상위어이면
        직전 PROPOSE의 해당 entity를 흡수.
        """
        cur_idx = current_proposal['idx']
        cur_ent_texts = [e['text'] for e in current_proposal['entities']]
        
        for p in reversed(proposals):
            if p is current_proposal:
                continue
            if p['status'] not in ('pending', 'confirmed'):
                continue
            if (cur_idx - p['idx']) > lookback:
                break
            
            keep_ents = []
            for prev_ent in p['entities']:
                if prev_ent['type'] in META_TYPES:
                    keep_ents.append(prev_ent)
                    continue
                
                # type이 다르면 흡수 안 함 (FOOD vs LOC 등)
                same_type_cur_texts = [
                    e['text'] for e in current_proposal['entities']
                    if e['type'] == prev_ent['type']
                       or (prev_ent['type'] == 'ACTIVITY' and e['type'] == 'LOC')
                       or (prev_ent['type'] == 'LOC' and e['type'] == 'ACTIVITY')
                ]
                
                if same_type_cur_texts and is_subsumed_by(prev_ent['text'], same_type_cur_texts):
                    continue  # 흡수
                keep_ents.append(prev_ent)
            
            if not keep_ents:
                p['status'] = 'absorbed'
            else:
                p['entities'] = keep_ents
            
            # 현재 메시지의 entities에서도 중복(자기 자신 흡수) 제거
            seen = set()
            unique_cur_ents = []
            for e in current_proposal['entities']:
                key = (e['type'], e['text'])
                if key not in seen:
                    seen.add(key)
                    unique_cur_ents.append(e)
            current_proposal['entities'] = unique_cur_ents
            
            # 현재 메시지 안에서도 일반→구체 흡수 처리
            # (예: "닭강정"+"만석닭강정"이 한 메시지에 같이 있을 때)
            cur_ents = current_proposal['entities']
            keep_self = []
            for e1 in cur_ents:
                if e1['type'] in META_TYPES:
                    keep_self.append(e1)
                    continue
                # 다른 entity가 e1을 흡수하는지
                others = [e2['text'] for e2 in cur_ents 
                          if e2 is not e1 
                          and (e2['type'] == e1['type']
                               or (e1['type'] == 'ACTIVITY' and e2['type'] == 'LOC')
                               or (e1['type'] == 'LOC' and e2['type'] == 'ACTIVITY'))]
                if others and is_subsumed_by(e1['text'], others):
                    continue
                keep_self.append(e1)
            current_proposal['entities'] = keep_self
            
            return
    
    for i, msg in enumerate(messages):
        if not msg.get('in_travel_span'):
            continue
        
        intent = msg.get('intent')
        ents = msg.get('entities', [])
        text = msg['text']
        
        day_kw = detect_day_keyword(text, current_day)
        if day_kw is not None:
            if day_kw == -1:
                current_day = 'LAST'
            else:
                current_day = day_kw
            
            # AGREE/CONFIRM에 day 키워드가 있으면 → 가장 최근 PROPOSE의 day_hint 재할당
            # 예: idx 121 경포대 PROPOSE → idx 125 "셋째날에 가보자" AGREE
            #     → 경포대를 day 3로 재배치
            if intent in ('AGREE', 'CONFIRM'):
                target_day = current_day
                # 직전 lookback 내 가장 가까운 confirmed/pending PROPOSE 1개만
                for p in reversed(proposals):
                    if (i - p['idx']) > lookback:
                        break
                    if p['status'] in ('pending', 'confirmed'):
                        p['day_hint'] = target_day
                        break  # 하나만 재할당
            
            # 새 day 키워드(예: "마지막 날") 만나면 → 직전 pending PROPOSE는 암묵적 confirmed
            # (다음 토픽으로 넘어가기 전 명시적 거절 없으면 동의로 간주)
            if intent in ('QUERY', 'PROPOSE', 'AGREE', 'CONFIRM'):
                for p in proposals:
                    if p['status'] == 'pending' and (i - p['idx']) <= lookback:
                        p['status'] = 'confirmed'
                        p['response_text'] = '(implicit-confirm-by-topic-shift)'
        
        time_str = detect_message_time(text, ents)
        
        # 숙소 위치 의논 맥락 LOC 필터링
        span_idx = msg_to_span_idx.get(id(msg), -1)
        filtered_ents = []
        for ent in ents:
            if ent['type'] == 'LOC' and span_idx >= 0:
                if is_lodging_context_loc(span_idx, in_span_msgs, ent['text']):
                    continue
            filtered_ents.append(ent)
        ents = filtered_ents
        
        if intent == 'PROPOSE' and ents:
            has_agree_token = any(tok in text for tok in AGREE_TOKENS)
            
            new_proposal = {
                'idx': i,
                'msg': msg,
                'entities': ents,
                'status': 'pending',
                'day_hint': current_day,
                'time': time_str,
                'response_text': None,
            }
            proposals.append(new_proposal)
            
            if has_agree_token:
                new_proposal['status'] = 'confirmed'
                new_proposal['response_text'] = '(self-agree)'
                # 직전 일반 PROPOSE 흡수 시도
                absorb_general_by_specific(new_proposal)
                # 그 후 블록 confirm
                confirm_block(i, text)
        
        elif intent in ('AGREE', 'CONFIRM') and ents:
            confirm_block(i, text)
            
            non_meta_ents = [e for e in ents if e['type'] not in META_TYPES]
            new_meta_ents = [e for e in ents if e['type'] in META_TYPES]
            
            has_agree_token = any(tok in text for tok in AGREE_TOKENS)
            
            # 일반 entity는 동의 토큰(ㄱㄱ/ㅇㅋ/콜)이 있을 때만 새 정보로 추가
            # (구체화형 vs 부연설명형 구분: 부연설명엔 동의 토큰 없음)
            # 예: "닭강정은 무조건 만석닭강정 ㄱㄱ" → 추가 (구체화)
            #     "서울에서 강릉까지 금방이잖아" → 추가 안 함 (부연)
            if non_meta_ents and has_agree_token:
                new_proposal = {
                    'idx': i,
                    'msg': msg,
                    'entities': non_meta_ents,
                    'status': 'confirmed',
                    'day_hint': current_day,
                    'time': time_str,
                    'response_text': text,
                }
                proposals.append(new_proposal)
                absorb_general_by_specific(new_proposal)
            
            # 메타 entity는 항상 등록 (DURATION/DATE는 신뢰)
            if new_meta_ents:
                proposals.append({
                    'idx': i,
                    'msg': msg,
                    'entities': new_meta_ents,
                    'status': 'confirmed',
                    'day_hint': current_day,
                    'time': time_str,
                    'response_text': None,
                })
        
        elif intent in ('AGREE', 'CONFIRM'):
            confirm_block(i, text)
        
        elif intent in ('DISAGREE', 'CANCEL'):
            for p in reversed(proposals):
                if p['status'] in ('pending', 'confirmed') and (i - p['idx']) <= lookback:
                    p['status'] = 'cancelled'
                    p['response_text'] = text
                    break
    
    return proposals


# ============================================================
# 메타 정보 추출 (개선)
# ============================================================

def extract_meta(proposals, all_messages=None):
    """
    destination: 도시급 LOC 우선, sub-location/주소 제외
    빈도가 높은 도시급 LOC가 destination (모든 메시지 기준, intent 무관)
    """
    meta = {
        'destination': None,
        'duration': None,
        'start_date': None,
        'main_lodging': None,
        'main_transport': None,
    }
    
    # destination: 모든 메시지의 LOC entity 집계 (intent 무관)
    # PROPOSE에 한정하면 "강릉 가고 싶다"(OTHER) 같은 게 빠짐
    from collections import Counter
    city_counter = Counter()
    
    if all_messages:
        for m in all_messages:
            if not m.get('in_travel_span'):
                continue
            for e in m.get('entities', []):
                if e['type'] != 'LOC':
                    continue
                if is_address(e['text']):
                    continue
                if is_city_location(e['text']):
                    city_counter[e['text']] += 1
    
    # 가장 많이 언급된 도시급 LOC가 destination
    if city_counter:
        meta['destination'] = city_counter.most_common(1)[0][0]
    
    # 도시 후보 없으면 LODGING 텍스트에서 도시명 추출 시도
    if not meta['destination']:
        for p in proposals:
            for e in p['entities']:
                if e['type'] == 'LODGING':
                    # "강릉 ING 게스트하우스" → "강릉"
                    for kw in CITY_KEYWORDS:
                        if e['text'].startswith(kw):
                            meta['destination'] = kw
                            break
                    if meta['destination']:
                        break
            if meta['destination']:
                break
    
    # duration: 어디서든 추출 (confirmed 우선)
    for status_priority in ['confirmed', 'pending']:
        for p in proposals:
            if p['status'] != status_priority:
                continue
            for e in p['entities']:
                if e['type'] == 'DURATION':
                    norm = normalize_duration(e['text'])
                    # "도깨비" 같은 엉뚱한 게 DURATION으로 잡힌 건 제외
                    if 'nights' in norm or '당일' in e['text']:
                        meta['duration'] = norm['display']
                        break
            if meta['duration']:
                break
        if meta['duration']:
            break
    
    # date
    for status_priority in ['confirmed', 'pending']:
        for p in proposals:
            if p['status'] != status_priority:
                continue
            for e in p['entities']:
                if e['type'] == 'DATE' and not detect_day_keyword(e['text'], 1):
                    meta['start_date'] = normalize_date(e['text'])
                    break
            if meta['start_date']:
                break
        if meta['start_date']:
            break
    
    # 숙소 (마지막 confirmed 우선, "숙소" 같은 일반어 제외)
    for p in reversed(proposals):
        if p['status'] != 'confirmed':
            continue
        for e in p['entities']:
            if e['type'] == 'LODGING' and e['text'] not in ('숙소',):
                meta['main_lodging'] = e['text']
                break
        if meta['main_lodging']:
            break
    
    # 교통수단
    for p in proposals:
        if p['status'] != 'confirmed':
            continue
        for e in p['entities']:
            if e['type'] == 'TRANSPORT':
                meta['main_transport'] = e['text']
                break
        if meta['main_transport']:
            break
    
    return meta


# ============================================================
# Stage 6: 일자별 일정표 (개선)
# ============================================================

def is_meta_only(entities):
    return all(e['type'] in META_TYPES for e in entities)


def get_total_days(duration_str):
    """duration 문자열에서 총 day 수"""
    if not duration_str:
        return None
    m = re.search(r'(\d+)일', duration_str)
    if m:
        return int(m.group(1))
    if '당일' in duration_str:
        return 1
    return None


def build_schedule(proposals, total_days, destination=None, lodging=None, main_transport=None):
    """
    confirmed PROPOSE를 일자별 이벤트로 변환.
    
    개선: 
    - 일반어("숙소", "점심", "저녁", "카페" 등)는 location으로 안 넣음
    - 명시적 day 키워드(LAST=마지막날) 제대로 처리
    - destination/lodging/transport와 중복되는 entity는 일정에서 제외
    """
    # ============================================================
    # 전처리: 일반 ACTIVITY/FOOD + 직후 구체 PLACE 묶기
    # 예: idx 96 "점심 먹자" + idx 97 "중앙시장 가자"
    #     → 중앙시장 (memo: 점심 식사)
    # ============================================================
    GENERIC_ACTS_FOR_MERGE = {'점심', '저녁', '아침', '식사', '카페'}
    SEPARATION_WORDS = ['그리고', '그담', '그담에', '먹고', '하고', '갔다가']
    
    confirmed_sorted = sorted(
        [p for p in proposals if p['status'] == 'confirmed'],
        key=lambda p: p['idx']
    )
    
    merged_prev_idx = set()
    merge_into_next = {}  # next의 idx → 흡수할 prev entities
    
    for i in range(len(confirmed_sorted) - 1):
        prev = confirmed_sorted[i]
        
        if prev['idx'] in merged_prev_idx:
            continue
        
        # prev가 PLACE 없이 일반 ACTIVITY/FOOD만 갖고 있는지
        prev_has_place = any(
            CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING')
            for e in prev['entities']
        )
        if prev_has_place:
            continue
        
        prev_act_food = [
            e for e in prev['entities']
            if CATEGORY_MAP.get(e['type']) in ('ACTIVITY', 'FOOD')
        ]
        if not prev_act_food:
            continue
        if not all(e['text'] in GENERIC_ACTS_FOR_MERGE or len(e['text']) <= 2
                    for e in prev_act_food):
            continue
        
        # 다음 PLACE/LODGING/FOOD 가진 PROPOSE 찾기 (최대 3턴 이내)
        nxt = None
        for j in range(i + 1, len(confirmed_sorted)):
            cand = confirmed_sorted[j]
            if cand['idx'] - prev['idx'] > 3:
                break
            if cand['day_hint'] != prev['day_hint']:
                break
            
            cand_has_specific = any(
                CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING', 'FOOD')
                and e['text'] not in GENERIC_ACTS_FOR_MERGE
                and e['text'] not in (destination, lodging, main_transport)
                for e in cand['entities']
            )
            if cand_has_specific:
                nxt = cand
                break
        
        if nxt is None:
            continue
        
        # next 텍스트에 시간 순서 단어 있으면 합치지 않음
        if any(sw in nxt['msg']['text'] for sw in SEPARATION_WORDS):
            continue
        
        merged_prev_idx.add(prev['idx'])
        merge_into_next[nxt['idx']] = prev_act_food
    
    # 흡수된 prev는 absorbed로 표시
    for p in proposals:
        if p['idx'] in merged_prev_idx:
            p['status'] = 'absorbed'
    
    # next에 흡수된 entity 병합 (실제 객체 수정)
    for p in proposals:
        if p['idx'] in merge_into_next:
            p['entities'] = list(p['entities']) + merge_into_next[p['idx']]
    
    # ============================================================
    # 본 처리
    # ============================================================
    confirmed = [p for p in proposals if p['status'] == 'confirmed']
    schedule_props = [p for p in confirmed if not is_meta_only(p['entities'])]
    # absorbed는 이미 confirmed 아니므로 자동 제외됨
    
    # 일반명사 (location 자리에 들어가면 어색한 것)
    GENERIC_NOUNS = {'숙소', '점심', '저녁', '아침', '카페', '식사', '시장'}
    
    # destination/lodging/transport과 중복되는 텍스트 (일정에서 제외)
    dedup_set = set()
    if destination:
        dedup_set.add(destination)
    if lodging:
        dedup_set.add(lodging)
    if main_transport:
        dedup_set.add(main_transport)
    
    days_data = defaultdict(list)
    last_time_min = -1
    inferred_day = 1
    
    for p in schedule_props:
        # day 결정
        if p['day_hint'] == 'LAST':
            day = total_days if total_days else 999
        elif isinstance(p['day_hint'], int) and p['day_hint'] > 1:
            day = p['day_hint']
            inferred_day = day
        else:
            # 시간 흐름으로 추정 (시간이 거꾸로 가면 다음 day)
            t_min = time_to_minutes(p['time'])
            if t_min < last_time_min - 60 and t_min != 9999:
                inferred_day += 1
            day = inferred_day
            if t_min != 9999:
                last_time_min = t_min
        
        # 메시지 내 entity들을 분류
        valid_ents = []  # 일정에 들어갈 후보들
        for ent in p['entities']:
            cat = CATEGORY_MAP.get(ent['type'])
            if not cat:
                continue
            
            ent_text = ent['text']
            
            # 일반명사 LOC/LODGING은 제외 (예: "숙소", "시장" 단독)
            # ACTIVITY나 FOOD는 일반어여도 활동을 나타내므로 유지
            if ent_text in GENERIC_NOUNS and ent['type'] in ('LOC', 'LODGING'):
                continue
            
            if ent_text in dedup_set:
                continue
            if ent['type'] == 'LOC' and is_address(ent_text):
                continue
            if ent['type'] == 'LOC':
                role = detect_loc_role(p['msg']['text'], ent_text)
                if role != 'event':
                    continue
            
            valid_ents.append((ent, cat))
        
        if not valid_ents:
            continue
        
        # "X 갈거면 Y도", "X도 Y도", "X에서 Y하고 Z도" 같이 별개 활동 나열 패턴 검사
        # 이런 경우 LOC끼리 묶지 않고 각각 별개 이벤트로
        msg_text = p['msg']['text']
        is_multi_activity = bool(re.search(r'갈거면.*도\s', msg_text)) or \
                             bool(re.search(r'도\s+가자', msg_text)) or \
                             bool(re.search(r'도\s+들리', msg_text))
        
        # LOC/LODGING이 있으면 그게 메인, 나머지(FOOD/ACTIVITY)는 memo로
        # 단, multi_activity 패턴이면 묶지 않고 각각 별도 이벤트
        place_ents = [(e, c) for e, c in valid_ents if c in ('PLACE', 'LODGING')]
        food_ents = [(e, c) for e, c in valid_ents if c == 'FOOD']
        activity_ents = [(e, c) for e, c in valid_ents if c == 'ACTIVITY']
        transport_ents = [(e, c) for e, c in valid_ents if c == 'TRANSPORT']
        
        if is_multi_activity:
            # 별도 이벤트로 추가 (묶지 않음)
            for ent, cat in valid_ents:
                event = {
                    'time': p['time'],
                    'location': ent['text'],
                    'category': cat,
                    'source_text': p['msg']['text'],
                }
                days_data[day].append(event)
            continue
        
        # memo 문자열 만들기
        memo_parts = []
        for e, _ in food_ents:
            memo_parts.append(f"{e['text']} 먹기")
        for e, _ in activity_ents:
            txt = e['text']
            if txt in ('점심', '저녁', '아침', '식사'):
                memo_parts.append(f"{txt} 먹기")
            elif txt == '카페':
                memo_parts.append("카페")
            elif txt == '산책':
                memo_parts.append("산책")
            else:
                memo_parts.append(txt)
        memo = ', '.join(memo_parts) if memo_parts else None
        
        if place_ents:
            # PLACE/LODGING 있으면 그걸 메인 location으로, FOOD/ACTIVITY는 memo
            for ent, cat in place_ents:
                event = {
                    'time': p['time'],
                    'location': ent['text'],
                    'category': cat,
                    'source_text': p['msg']['text'],
                }
                if memo:
                    event['memo'] = memo
                days_data[day].append(event)
        elif food_ents:
            # PLACE 없지만 FOOD가 있으면 FOOD를 메인으로, 일반 ACTIVITY는 memo로
            # (예: idx 169 "초당할머니순두부 ㄱㄱ" + idx 166 "점심" 흡수)
            #     → 초당할머니순두부 (memo: 점심 식사)
            activity_memo_parts = []
            for e, _ in activity_ents:
                txt = e['text']
                if txt in ('점심', '저녁', '아침', '식사'):
                    activity_memo_parts.append(f"{txt} 식사" if txt != '식사' else '식사')
                else:
                    activity_memo_parts.append(txt)
            act_memo = ', '.join(activity_memo_parts) if activity_memo_parts else None
            
            for ent, cat in food_ents:
                event = {
                    'time': p['time'],
                    'location': ent['text'],
                    'category': cat,
                    'source_text': p['msg']['text'],
                }
                if act_memo:
                    event['memo'] = act_memo
                days_data[day].append(event)
        else:
            # PLACE도 FOOD도 없으면 ACTIVITY가 메인
            for ent, cat in valid_ents:
                loc_text = ent['text']
                if cat == 'ACTIVITY' and loc_text in ('점심', '저녁', '아침', '식사'):
                    loc_text = f"{loc_text} 식사" if loc_text != '식사' else '식사'
                
                event = {
                    'time': p['time'],
                    'location': loc_text,
                    'category': cat,
                    'source_text': p['msg']['text'],
                }
                days_data[day].append(event)
    
    # 999(LAST 임시값)는 마지막 실제 day로 변환
    if 999 in days_data and total_days:
        days_data[total_days].extend(days_data.pop(999))
    
    sorted_days = sorted(days_data.keys())
    
    days_list = []
    seen_locations_per_day = {}
    for d_idx, day in enumerate(sorted_days, 1):
        events = days_data[day]
        events.sort(key=lambda e: time_to_minutes(e['time']))
        
        # 같은 day 내 중복 location 제거
        unique_events = []
        seen = set()
        for e in events:
            key = (e['location'], e['category'])
            if key not in seen:
                seen.add(key)
                unique_events.append(e)
        
        days_list.append({
            'day': d_idx,
            'events': unique_events,
        })
    
    return days_list


def build_pending_cancelled(proposals, dedup_set):
    pending = []
    cancelled = []
    
    GENERIC_NOUNS = {'숙소', '점심', '저녁', '아침', '카페', '식사', '시장'}
    
    for p in proposals:
        for ent in p['entities']:
            if ent['type'] in META_TYPES:
                continue
            cat = CATEGORY_MAP.get(ent['type'])
            if not cat:
                continue
            ent_text = ent['text']
            if ent_text in GENERIC_NOUNS or ent_text in dedup_set:
                continue
            if ent['type'] == 'LOC' and is_address(ent_text):
                continue
            if ent['type'] == 'LOC':
                role = detect_loc_role(p['msg']['text'], ent_text)
                if role != 'event':
                    continue
            
            item = {
                'category': cat,
                'location': ent_text,
                'source_text': p['msg']['text'],
            }
            if p['status'] == 'pending':
                pending.append(item)
            elif p['status'] == 'cancelled':
                item['cancel_reason'] = p.get('response_text', '')
                cancelled.append(item)
    
    return pending, cancelled


# ============================================================
# 메인 파이프라인
# ============================================================

def split_into_chats(results):
    if results and results[0].get('source'):
        chats = defaultdict(list)
        for r in results:
            chats[r['source']].append(r)
        return list(chats.items())
    return [('chat_1', results)]


def process(pred_results, verbose=False):
    chats = split_into_chats(pred_results)
    summaries = []
    
    for chat_id, chat_msgs in chats:
        if verbose:
            print(f"\n[{chat_id}] 메시지 {len(chat_msgs)}개 처리 중...")
        
        proposals = assemble_events(chat_msgs)
        meta = extract_meta(proposals, all_messages=chat_msgs)
        departure, return_via = extract_departure_return(chat_msgs)
        total_days = get_total_days(meta['duration'])
        
        dedup_set = {meta['destination'], meta['main_lodging'], meta['main_transport']}
        dedup_set.discard(None)
        
        days = build_schedule(proposals, total_days, 
                              destination=meta['destination'],
                              lodging=meta['main_lodging'],
                              main_transport=meta['main_transport'])
        pending, cancelled = build_pending_cancelled(proposals, dedup_set)
        
        summary = {
            'chat_id': chat_id,
            'destination': meta['destination'],
            'duration': meta['duration'],
            'start_date': meta['start_date'],
            'lodging': meta['main_lodging'],
            'transport': meta['main_transport'],
            'departure': departure,
            'return_via': return_via,
            'days': days,
            'pending': pending,
            'cancelled': cancelled,
            'stats': {
                'n_messages': len(chat_msgs),
                'n_in_span': sum(1 for m in chat_msgs if m.get('in_travel_span')),
                'n_proposals': len(proposals),
                'n_confirmed': sum(1 for p in proposals if p['status'] == 'confirmed'),
                'n_cancelled': sum(1 for p in proposals if p['status'] == 'cancelled'),
            },
        }
        
        if verbose:
            print(f"  목적지: {meta['destination']}, 기간: {meta['duration']}")
            if departure:
                print(f"  출발: {departure['location']} {departure['time'] or ''}")
            if return_via:
                print(f"  귀환 경유: {', '.join(return_via)}")
            print(f"  PROPOSE {summary['stats']['n_proposals']}개 → "
                  f"확정 {summary['stats']['n_confirmed']}, "
                  f"취소 {summary['stats']['n_cancelled']}")
            for day in days:
                print(f"  Day {day['day']}: {len(day['events'])}개 이벤트")
        
        summaries.append(summary)
    
    return summaries


def format_human_readable(summary):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"여행 일정표 [{summary['chat_id']}]")
    lines.append("=" * 60)
    
    if summary['destination']:
        lines.append(f"📍 목적지   : {summary['destination']}")
    if summary['duration']:
        lines.append(f"📅 기간     : {summary['duration']}")
    if summary['start_date']:
        lines.append(f"🗓  날짜     : {summary['start_date']}")
    if summary['transport']:
        lines.append(f"🚗 이동     : {summary['transport']}")
    if summary['lodging']:
        lines.append(f"🏨 숙소     : {summary['lodging']}")
    
    if summary.get('departure'):
        dep = summary['departure']
        time_str = f" ({dep['time']})" if dep.get('time') else ""
        lines.append(f"🚉 출발지   : {dep['location']}{time_str}")
    
    if summary.get('return_via'):
        lines.append(f"↩️  귀환 경유 : {', '.join(summary['return_via'])}")
    
    lines.append("")
    
    for day in summary['days']:
        lines.append(f"━━━ Day {day['day']} ━━━")
        for ev in day['events']:
            time = ev['time'] if ev['time'] else '  -  '
            cat = ev['category']
            loc = ev['location']
            memo = ev.get('memo')
            line = f"  {time:>5s}  [{cat:8s}] {loc}"
            if memo:
                line += f"  ({memo})"
            lines.append(line)
        lines.append("")
    
    if summary['pending']:
        lines.append(f"⏳ 미확정 ({len(summary['pending'])}건)")
        for p in summary['pending'][:5]:
            lines.append(f"   - [{p['category']}] {p['location']}")
        lines.append("")
    
    if summary['cancelled']:
        lines.append(f"❌ 취소됨 ({len(summary['cancelled'])}건)")
        for c in summary['cancelled'][:5]:
            reason = (c.get('cancel_reason', '') or '')[:25]
            lines.append(f"   - [{c['category']}] {c['location']}  ({reason})")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--output', default='summary.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    print(f"입력 로드: {args.pred}")
    with open(args.pred, encoding='utf-8') as f:
        pred = json.load(f)
    print(f"  메시지 {len(pred)}개")
    
    summaries = process(pred, verbose=args.verbose)
    print(f"\n채팅 {len(summaries)}개 처리 완료\n")
    
    for summary in summaries:
        print(format_human_readable(summary))
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 일정표 저장: {args.output}")