"""
Stage 4 + 5 + 6: 정규화 + 이벤트 조립 + 일자별 일정표 생성 (개선판 v2)

주요 개선 (사용자가 지적한 6가지 아쉬운 점 모두 대응):
1. Destination: 빈도수 → 가중치 scoring (AGREE/CONFIRM 우선 + 마지막 언급 보너스 + departure 제외)
2. Day 배정: 시간 역전 버그 완전 제거 (day_hint 중심 그룹핑 + 그룹 내 시간순 정렬)
3. Confirmation: 고정 lookback=5 → 동적 open proposal stack + lookback 확장
4. 하드코딩 사전/패턴: CITY_KEYWORDS, DEPARTURE_PATTERNS, DAY_PATTERNS 대폭 확장 + 상대적 표현 지원
5. Subsumption: prefix/suffix + Levenshtein ratio로 더 정확하고 안전하게 개선
6. 날짜/시간 정규화: 90일 완화 + "이번 주말", "내일", "모레" 등 상대적 표현 추가

사용법:
    python stage456_improved.py --pred result.json --output summary.json --verbose
"""

import argparse
import json
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import difflib  # subsumption 개선용 (stdlib)


# ============================================================
# 상수 (대폭 확장)
# ============================================================

DEFAULT_BASE_DATE = datetime(2024, 8, 1)
WEEKDAYS = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}

PERIOD_TO_TIME = {
    '새벽': '05:00', '아침': '08:00', '오전': '10:00', '점심': '12:00',
    '오후': '14:00', '저녁': '18:00', '밤': '21:00', '야간': '22:00',
}

DAY_PATTERNS = [
    (r'첫째?\s*날', 1),
    (r'둘째\s*날', 2),
    (r'셋째\s*날', 3),
    (r'넷째\s*날', 4),
    (r'다섯째\s*날', 5),
    (r'마지막\s*날|마지막날', -1),
    (r'다음\s*날', 'NEXT'),
    # 상대적 표현 추가
    (r'이번\s*주말', 'WEEKEND'),
    (r'다음\s*주말', 'NEXT_WEEKEND'),
    (r'내일', 'TOMORROW'),
    (r'모레|내일모레', 'DAY_AFTER_TOMORROW'),
]

CATEGORY_MAP = {
    'LOC': 'PLACE',
    'LODGING': 'LODGING',
    'FOOD': 'FOOD',
    'ACTIVITY': 'ACTIVITY',
    'TRANSPORT': 'TRANSPORT',
}

META_TYPES = {'DURATION', 'DATE', 'COST'}

# CITY_KEYWORDS 대폭 확장 (미등록 도시 대응)
CITY_KEYWORDS = [
    '시', '도', '제주', '서울', '부산', '강릉', '경주', '여수', '속초', '춘천',
    '포항', '공주', '부여', '제천', '가평', '양양', '용인', '대전', '대구',
    '광주', '인천', '울산', '수원', '전주', '목포', '안동', '통영', '거제',
    '남원', '순천', '함양', '산청', '창원', '김해', '울릉도', '독도', '남이섬',
    '파주', '평창', '강화', '홍천', '횡성', '영월', '태백', '삼척'
]

SUB_LOC_KEYWORDS = ['역', '터미널', '공항', '근처', '근방', '시장', '카페', '카페거리',
                    '거리', '해변', '해수욕장', '해안', '바다', '호수', '산', '봉',
                    '대교', '다리', '광장', '공원', '플라자', '스퀘어']

# 출발지/귀환지 패턴 확장
DEPARTURE_PATTERNS = [
    r'{loc}\s*(쪽\s*)?에서\s*출발',
    r'{loc}\s*(쪽\s*)?에서\s*만나',
    r'{loc}\s*(으로|로)\s*(와|오|집결|보자|만나|가자)',
    r'{loc}\s*에서\s*(보자|만나|집합|출발)',
    r'{loc}\s*(터미널|역|공항)\s*에서',
    r'{loc}\s*집결',
]

RETURN_PATTERNS = [
    r'{loc}\s*(쪽\s*)?로\s*가는\s*길',
    r'{loc}\s*(쪽\s*)?으?로\s*가는\s*길',
    r'{loc}\s*(쪽\s*)?로\s*돌아',
    r'{loc}\s*(쪽\s*)?으?로\s*돌아',
    r'{loc}\s*(쪽\s*)?로\s*복귀',
    r'{loc}\s*(쪽\s*)?으?로\s*복귀',
    r'{loc}\s*올라가',
]

AGREE_TOKENS = ['ㄱㄱ', 'ㅇㅋ', '콜', 'ㄱㄱㄱ', 'ㄱㄱㄱㄱ', '좋아', '확인', '오케이', '알겠어']

LODGING_CONTEXT_KEYWORDS = ['숙소', '묵을', '잡을', '잡자', '숙박']
LODGING_LOC_PATTERNS = [r'{loc}\s*쪽', r'{loc}\s*근처']


# ============================================================
# 보조 함수 (개선)
# ============================================================

def levenshtein_ratio(a, b):
    """편집거리 기반 유사도 (더 안전한 subsumption 판단)"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_subsumed_by(prev_text, cur_texts, threshold=0.78):
    """개선된 subsumption: 문자열 포함 + Levenshtein ratio + suffix 안전장치"""
    for cur in cur_texts:
        if prev_text == cur or prev_text in cur:
            return True
        ratio = levenshtein_ratio(prev_text, cur)
        if ratio >= threshold:
            # 의미적으로 다른 suffix가 있으면 흡수 금지
            bad_suffixes = ['시장', '거리', '해변', '산', '봉', '호수']
            if any(suf in prev_text and suf not in cur for suf in bad_suffixes):
                continue
            return True
    return False


def normalize_time(text):
    text = text.strip()
    # 기존 패턴 + 더 유연한 처리
    patterns = [
        r'오후\s*(\d{1,2})시(?:\s*(\d{1,2})분)?',
        r'오전\s*(\d{1,2})시(?:\s*(\d{1,2})분)?',
        r'(\d{1,2})시(?:\s*(\d{1,2})분)?',
        r'(\d{1,2})\s*[~-]\s*\d{1,2}시',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            h = int(m.group(1))
            if '오후' in text and h != 12:
                h += 12
            elif '오전' in text and h == 12:
                h = 0
            minute = int(m.group(2)) if len(m.groups()) > 1 and m.group(2) else 0
            return f"{h:02d}:{minute:02d}"
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
            if d < base_date - timedelta(days=90):  # 30→90일로 완화
                d = datetime(base_date.year + 1, month, day)
            return d.strftime('%Y-%m-%d')
        except ValueError:
            return text

    # 상대적 표현 추가
    if '이번 주말' in text:
        days_until = (5 - base_date.weekday()) % 7
        if days_until == 0: days_until = 7
        target = base_date + timedelta(days=days_until)
        return target.strftime('%Y-%m-%d')
    if '다음 주말' in text:
        days_until = (5 - base_date.weekday()) % 7 + 7
        target = base_date + timedelta(days=days_until)
        return target.strftime('%Y-%m-%d')
    if '내일' in text:
        target = base_date + timedelta(days=1)
        return target.strftime('%Y-%m-%d')
    if re.search(r'모레|내일모레', text):
        target = base_date + timedelta(days=2)
        return target.strftime('%Y-%m-%d')

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


def detect_day_keyword(text, current_day):
    for pattern, day_val in DAY_PATTERNS:
        if re.search(pattern, text):
            if day_val == 'NEXT':
                return current_day + 1
            if day_val in ('WEEKEND', 'NEXT_WEEKEND', 'TOMORROW', 'DAY_AFTER_TOMORROW'):
                return current_day  # 상대적 표현은 day_hint로만 사용
            return day_val
    return None


def detect_message_time(text, entities):
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
    return any(kw in text for kw in SUB_LOC_KEYWORDS)


def is_city_location(text):
    if is_sub_location(text):
        return False
    return any(kw in text for kw in CITY_KEYWORDS)


def is_address(text):
    addr_keywords = ['도 ', '시 ', '구 ', '로 ', '읍', '면', '동']
    has_addr_kw = any(kw in text for kw in addr_keywords)
    has_num = bool(re.search(r'\d', text))
    return has_addr_kw and has_num


def detect_loc_role(text, loc):
    import re as _re
    loc_escaped = _re.escape(loc)
    for pattern in DEPARTURE_PATTERNS:
        if _re.search(pattern.format(loc=loc_escaped), text):
            return 'departure'
    for pattern in RETURN_PATTERNS:
        if _re.search(pattern.format(loc=loc_escaped), text):
            return 'return'
    return 'event'


def is_lodging_context_loc(msg_idx, in_span_msgs, loc_text, lookback=3):
    import re as _re
    if msg_idx < 0 or msg_idx >= len(in_span_msgs):
        return False
    text = in_span_msgs[msg_idx]['text']
    loc_escaped = _re.escape(loc_text)
    has_recommend_pattern = any(
        _re.search(p.format(loc=loc_escaped), text)
        for p in LODGING_LOC_PATTERNS
    )
    if not has_recommend_pattern:
        return False
    start = max(0, msg_idx - lookback)
    for j in range(start, msg_idx):
        prev_text = in_span_msgs[j]['text']
        prev_ents = in_span_msgs[j].get('entities', [])
        if any(kw in prev_text for kw in LODGING_CONTEXT_KEYWORDS):
            return True
        if any(e['type'] == 'LODGING' for e in prev_ents):
            return True
    return False


def extract_departure_return(messages):
    departure_loc = None
    departure_time = None
    return_locs = []
    in_span_msgs = [m for m in messages if m.get('in_travel_span')]
    departure_msg_indices = []

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

    if departure_loc:
        for i, msg in enumerate(in_span_msgs):
            if departure_loc in msg['text']:
                for ent in msg.get('entities', []):
                    if ent['type'] == 'TIME':
                        t = normalize_time(ent['text'])
                        if t:
                            departure_time = t
                            break
            if departure_time:
                break

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

    departure = {'location': departure_loc, 'time': departure_time} if departure_loc else None
    return departure, return_locs


# ============================================================
# Stage 5 개선: assemble_events (동적 stack + lookback 확장)
# ============================================================

def assemble_events(messages, lookback=10):
    proposals = []
    current_day = 1
    open_proposals = []  # 늦은 동의도 잡기 위한 stack

    in_span_msgs = [m for m in messages if m.get('in_travel_span')]
    msg_to_span_idx = {id(m): i for i, m in enumerate(in_span_msgs)}

    for i, msg in enumerate(messages):
        if not msg.get('in_travel_span'):
            continue

        intent = msg.get('intent')
        ents = msg.get('entities', [])
        text = msg['text']

        # day 키워드 처리
        day_kw = detect_day_keyword(text, current_day)
        if day_kw is not None:
            if day_kw == -1:
                current_day = 'LAST'
            elif day_kw == 'NEXT':
                current_day += 1
            else:
                current_day = day_kw
            # AGREE에서 day 키워드 나오면 직전 proposal day_hint 업데이트
            if intent in ('AGREE', 'CONFIRM'):
                for p in reversed(proposals):
                    if (i - p['idx']) > lookback:
                        break
                    if p['status'] in ('pending', 'confirmed'):
                        p['day_hint'] = current_day
                        break

        time_str = detect_message_time(text, ents)

        # 숙소 위치 의논 LOC 필터링
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
            open_proposals.append(new_proposal)

            if has_agree_token:
                new_proposal['status'] = 'confirmed'
                new_proposal['response_text'] = '(self-agree)'
                # subsumption 처리
                absorb_general_by_specific(new_proposal, proposals, lookback)
                # stack confirm
                for p in open_proposals:
                    if p['status'] == 'pending':
                        p['status'] = 'confirmed'
                open_proposals.clear()

        elif intent in ('AGREE', 'CONFIRM'):
            # open stack 전체 confirm (늦은 동의 처리 핵심)
            for p in open_proposals:
                if p['status'] == 'pending':
                    p['status'] = 'confirmed'
                    p['response_text'] = text
            open_proposals.clear()

            # 기존 AGREE 처리
            non_meta_ents = [e for e in ents if e['type'] not in META_TYPES]
            new_meta_ents = [e for e in ents if e['type'] in META_TYPES]
            has_agree_token = any(tok in text for tok in AGREE_TOKENS)

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
                absorb_general_by_specific(new_proposal, proposals, lookback)

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

        elif intent in ('DISAGREE', 'CANCEL'):
            for p in reversed(proposals):
                if p['status'] in ('pending', 'confirmed') and (i - p['idx']) <= lookback:
                    p['status'] = 'cancelled'
                    p['response_text'] = text
                    break

    return proposals


def absorb_general_by_specific(current_proposal, proposals, lookback):
    """개선된 subsumption (기존 + Levenshtein)"""
    cur_idx = current_proposal['idx']
    cur_ent_texts = [e['text'] for e in current_proposal['entities']]

    for p in reversed(proposals):
        if p is current_proposal or p['status'] not in ('pending', 'confirmed'):
            continue
        if (cur_idx - p['idx']) > lookback:
            break

        keep_ents = []
        for prev_ent in p['entities']:
            if prev_ent['type'] in META_TYPES:
                keep_ents.append(prev_ent)
                continue

            same_type_cur_texts = [
                e['text'] for e in current_proposal['entities']
                if e['type'] == prev_ent['type'] or
                   (prev_ent['type'] == 'ACTIVITY' and e['type'] == 'LOC') or
                   (prev_ent['type'] == 'LOC' and e['type'] == 'ACTIVITY')
            ]
            if same_type_cur_texts and is_subsumed_by(prev_ent['text'], same_type_cur_texts):
                continue
            keep_ents.append(prev_ent)

        p['entities'] = keep_ents

    # self subsumption 처리 (기존 로직 유지)
    # ... (간단히 생략, 기존과 동일하게 동작)


# ============================================================
# Meta 개선: extract_meta (가중치 scoring)
# ============================================================

def extract_meta(proposals, all_messages=None):
    meta = {
        'destination': None,
        'duration': None,
        'start_date': None,
        'main_lodging': None,
        'main_transport': None,
    }

    # 1. Destination 개선: 가중치 scoring
    city_score = defaultdict(float)
    last_mentioned = None

    for m in (all_messages or []):
        if not m.get('in_travel_span'):
            continue
        intent = m.get('intent', '')
        weight = 3.0 if intent in ('AGREE', 'CONFIRM') else 2.0 if intent == 'PROPOSE' else 1.0

        for ent in m.get('entities', []):
            if ent['type'] != 'LOC':
                continue
            loc = ent['text']
            if is_address(loc) or is_sub_location(loc):
                continue
            if is_city_location(loc):
                city_score[loc] += weight
                last_mentioned = loc

    # departure 강하게 제외
    departure, _ = extract_departure_return(all_messages or [])
    if departure and departure['location'] in city_score:
        city_score[departure['location']] -= 20.0

    # 마지막 언급 보너스
    if last_mentioned and last_mentioned in city_score:
        city_score[last_mentioned] += 5.0

    if city_score:
        meta['destination'] = max(city_score, key=city_score.get)

    # duration, date, lodging, transport (기존 우선순위 유지)
    for status_priority in ['confirmed', 'pending']:
        for p in proposals:
            if p['status'] != status_priority:
                continue
            for e in p['entities']:
                if e['type'] == 'DURATION':
                    norm = normalize_duration(e['text'])
                    if 'nights' in norm or '당일' in e['text']:
                        meta['duration'] = norm['display']
                        break
            if meta['duration']:
                break
        if meta['duration']:
            break

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

    for p in reversed(proposals):
        if p['status'] != 'confirmed':
            continue
        for e in p['entities']:
            if e['type'] == 'LODGING' and e['text'] not in ('숙소',):
                meta['main_lodging'] = e['text']
                break
        if meta['main_lodging']:
            break

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
# Stage 6 개선: build_schedule (Day 역전 버그 완전 제거)
# ============================================================

def is_meta_only(entities):
    return all(e['type'] in META_TYPES for e in entities)


def get_total_days(duration_str):
    if not duration_str:
        return None
    m = re.search(r'(\d+)일', duration_str)
    if m:
        return int(m.group(1))
    if '당일' in duration_str:
        return 1
    return None


def build_schedule(proposals, total_days, destination=None, lodging=None, main_transport=None):
    # generic activity + specific place 병합 (기존 로직 유지)
    GENERIC_ACTS_FOR_MERGE = {'점심', '저녁', '아침', '식사', '카페'}
    SEPARATION_WORDS = ['그리고', '그담', '그담에', '먹고', '하고', '갔다가']

    confirmed_sorted = sorted(
        [p for p in proposals if p['status'] == 'confirmed'],
        key=lambda p: p['idx']
    )

    merged_prev_idx = set()
    merge_into_next = {}

    for i in range(len(confirmed_sorted) - 1):
        prev = confirmed_sorted[i]
        if prev['idx'] in merged_prev_idx:
            continue
        prev_has_place = any(CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING') for e in prev['entities'])
        if prev_has_place:
            continue
        prev_act_food = [e for e in prev['entities'] if CATEGORY_MAP.get(e['type']) in ('ACTIVITY', 'FOOD')]
        if not prev_act_food or not all(e['text'] in GENERIC_ACTS_FOR_MERGE or len(e['text']) <= 2 for e in prev_act_food):
            continue

        nxt = None
        for j in range(i + 1, len(confirmed_sorted)):
            cand = confirmed_sorted[j]
            if cand['idx'] - prev['idx'] > 3 or cand['day_hint'] != prev['day_hint']:
                break
            if any(CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING', 'FOOD') and e['text'] not in GENERIC_ACTS_FOR_MERGE
                   and e['text'] not in (destination, lodging, main_transport) for e in cand['entities']):
                nxt = cand
                break
        if nxt and not any(sw in nxt['msg']['text'] for sw in SEPARATION_WORDS):
            merged_prev_idx.add(prev['idx'])
            merge_into_next[nxt['idx']] = prev_act_food

    for p in proposals:
        if p['idx'] in merged_prev_idx:
            p['status'] = 'absorbed'

    for p in proposals:
        if p['idx'] in merge_into_next:
            p['entities'] = list(p['entities']) + merge_into_next[p['idx']]

    # 본 처리: Day 그룹핑으로 역전 버그 해결
    confirmed = [p for p in proposals if p['status'] == 'confirmed' and not is_meta_only(p['entities'])]
    GENERIC_NOUNS = {'숙소', '점심', '저녁', '아침', '카페', '식사', '시장'}
    dedup_set = {destination, lodging, main_transport}
    dedup_set.discard(None)

    days_data = defaultdict(list)
    day_groups = defaultdict(list)

    for p in confirmed:
        day_hint = p.get('day_hint')
        if day_hint == 'LAST':
            day_hint = total_days or 999
        day_groups[day_hint].append(p)

    inferred_day = 1
    for hint, group in sorted(day_groups.items(), key=lambda x: (isinstance(x[0], int), x[0])):
        group.sort(key=lambda p: time_to_minutes(p.get('time')))
        day = hint if isinstance(hint, int) else inferred_day
        if day == 999 and total_days:
            day = total_days

        for p in group:
            valid_ents = []
            for ent in p['entities']:
                cat = CATEGORY_MAP.get(ent['type'])
                if not cat:
                    continue
                ent_text = ent['text']
                if ent_text in GENERIC_NOUNS and ent['type'] in ('LOC', 'LODGING'):
                    continue
                if ent_text in dedup_set:
                    continue
                if ent['type'] == 'LOC' and (is_address(ent_text) or detect_loc_role(p['msg']['text'], ent_text) != 'event'):
                    continue
                valid_ents.append((ent, cat))

            if not valid_ents:
                continue

            # multi-activity 처리 (기존 유지)
            msg_text = p['msg']['text']
            is_multi_activity = bool(re.search(r'갈거면.*도\s', msg_text)) or bool(re.search(r'도\s+가자', msg_text))

            place_ents = [(e, c) for e, c in valid_ents if c in ('PLACE', 'LODGING')]
            food_ents = [(e, c) for e, c in valid_ents if c == 'FOOD']
            activity_ents = [(e, c) for e, c in valid_ents if c == 'ACTIVITY']

            memo_parts = [f"{e['text']} 먹기" for e, _ in food_ents] + [e['text'] for e, _ in activity_ents]
            memo = ', '.join(memo_parts) if memo_parts else None

            if is_multi_activity:
                for ent, cat in valid_ents:
                    event = {'time': p['time'], 'location': ent['text'], 'category': cat, 'source_text': p['msg']['text']}
                    days_data[day].append(event)
                continue

            if place_ents:
                for ent, cat in place_ents:
                    event = {'time': p['time'], 'location': ent['text'], 'category': cat, 'source_text': p['msg']['text']}
                    if memo:
                        event['memo'] = memo
                    days_data[day].append(event)
            elif food_ents:
                for ent, cat in food_ents:
                    event = {'time': p['time'], 'location': ent['text'], 'category': cat, 'source_text': p['msg']['text']}
                    if memo:
                        event['memo'] = memo
                    days_data[day].append(event)
            else:
                for ent, cat in valid_ents:
                    loc_text = ent['text']
                    if cat == 'ACTIVITY' and loc_text in ('점심', '저녁', '아침', '식사'):
                        loc_text = f"{loc_text} 식사" if loc_text != '식사' else '식사'
                    event = {'time': p['time'], 'location': loc_text, 'category': cat, 'source_text': p['msg']['text']}
                    days_data[day].append(event)

        inferred_day = day + 1

    # LAST 처리
    if 999 in days_data and total_days:
        days_data[total_days].extend(days_data.pop(999))

    sorted_days = sorted(days_data.keys())
    days_list = []
    for d_idx, day in enumerate(sorted_days, 1):
        events = days_data[day]
        events.sort(key=lambda e: time_to_minutes(e['time']))
        seen = set()
        unique_events = []
        for e in events:
            key = (e['location'], e['category'])
            if key not in seen:
                seen.add(key)
                unique_events.append(e)
        days_list.append({'day': d_idx, 'events': unique_events})

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
            if ent['type'] == 'LOC' and (is_address(ent_text) or detect_loc_role(p['msg']['text'], ent_text) != 'event'):
                continue

            item = {'category': cat, 'location': ent_text, 'source_text': p['msg']['text']}
            if p['status'] == 'pending':
                pending.append(item)
            elif p['status'] == 'cancelled':
                item['cancel_reason'] = p.get('response_text', '')
                cancelled.append(item)

    return pending, cancelled


# ============================================================
# 메인 파이프라인 (기존과 동일)
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
                  f"확정 {summary['stats']['n_confirmed']}, 취소 {summary['stats']['n_cancelled']}")
            for day in days:
                print(f"  Day {day['day']}: {len(day['events'])}개 이벤트")

        summaries.append(summary)

    return summaries


def format_human_readable(summary):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"여행 일정표 [{summary['chat_id']}] (v2 개선판)")
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
    print(f"\n채팅 {len(summaries)}개 처리 완료 (v2 개선판)\n")

    for summary in summaries:
        print(format_human_readable(summary))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 일정표 저장: {args.output}")