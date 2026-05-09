"""
TalkTrip-AI Schedule Builder v4.2
Rule-based Travel Schedule Pipeline
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
import difflib


# ============================================================
# 상수
# ============================================================

DEFAULT_BASE_DATE = datetime(2024, 8, 1)
WEEKDAYS = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}

PERIOD_TO_TIME = {
    '새벽': '05:00', '아침': '08:00', '오전': '10:00', '점심': '12:00',
    '오후': '14:00', '저녁': '18:00', '밤': '21:00', '야간': '22:00',
}

DEFAULT_TIME_SLOTS = {1: '10:00', 2: '12:00', 3: '15:00', 4: '18:00'}

DAY_PATTERNS = [
    (r'첫째?\s*날', 1), (r'둘째\s*날', 2), (r'셋째\s*날', 3),
    (r'넷째\s*날', 4), (r'다섯째\s*날', 5),
    (r'마지막\s*날|마지막날', -1), (r'다음\s*날', 'NEXT'),
    (r'이번\s*주말', 'WEEKEND'), (r'다음\s*주말', 'NEXT_WEEKEND'),
    (r'내일', 'TOMORROW'), (r'모레|내일모레', 'DAY_AFTER_TOMORROW'),
]

CATEGORY_MAP = {'LOC': 'PLACE', 'LODGING': 'LODGING', 'FOOD': 'FOOD',
                'ACTIVITY': 'ACTIVITY', 'TRANSPORT': 'TRANSPORT'}

META_TYPES = {'DURATION', 'DATE', 'COST'}

CITY_KEYWORDS = [
    '시', '도', '제주', '서울', '부산', '강릉', '경주', '여수', '속초', '춘천',
    '포항', '전주', '목포', '안동', '통영', '거제', '남원', '순천', '함양', '산청',
    '창원', '김해', '울릉도', '독도', '남이섬', '파주', '평창', '강화', '홍천', '횡성',
    '영월', '태백', '삼척', '대전', '대구', '광주', '인천', '울산', '수원', '용인',
    '가평', '양양', '제천', '공주', '부여', '강원', '경남', '경북'
]

SUB_LOC_KEYWORDS = [
    '역', '터미널', '공항', '근처', '시장', '카페', '카페거리', '거리', '해변',
    '해수욕장', '해안', '오름', '일출봉', '향교', '미술관', '박물관', '대관령',
    '델문도', '초당', '안목', '중앙시장', '플라자', '스퀘어', '봉', '산', '호수',
    '경포대', '주문진', '성산일출봉', '아부오름', '강릉향교', '강를시립미술관'
]

DEPARTURE_PATTERNS = [
    r'{loc}\s*(쪽\s*)?에서\s*출발', r'{loc}\s*(쪽\s*)?에서\s*만나',
    r'{loc}\s*(으로|로)\s*(와|오|집결|보자|만나)', r'{loc}\s*에서\s*(보자|만나|집합)',
    r'{loc}\s*(터미널|역|공항)\s*에서', r'{loc}\s*집결',
]

RETURN_PATTERNS = [
    r'{loc}\s*(쪽\s*)?로\s*가는\s*길', r'{loc}\s*(쪽\s*)?으?로\s*가는\s*길',
    r'{loc}\s*(쪽\s*)?로\s*돌아', r'{loc}\s*올라가',
]

AGREE_TOKENS = ['ㄱㄱ', 'ㅇㅋ', '콜', '좋아', '확인', '오케이', '알겠어', '그렇지', '좋다']

LODGING_CONTEXT_KEYWORDS = ['숙소', '묵을', '잡을', '잡자', '숙박']
LODGING_LOC_PATTERNS = [r'{loc}\s*쪽', r'{loc}\s*근처']


# ============================================================
# 보조 함수
# ============================================================

def levenshtein_ratio(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_subsumed_by(prev_text, cur_texts, threshold=0.75):
    for cur in cur_texts:
        if prev_text == cur or prev_text in cur:
            return True
        if levenshtein_ratio(prev_text, cur) >= threshold and len(prev_text) <= len(cur):
            bad = ['시장', '거리', '해변', '오름', '향교', '미술관', '박물관']
            if any(s in prev_text and s not in cur for s in bad):
                continue
            return True
    return False


def is_true_city_location(text):
    if any(kw in text for kw in SUB_LOC_KEYWORDS):
        return False
    return any(kw in text for kw in CITY_KEYWORDS)


def normalize_time(text):
    text = text.strip()
    m = re.search(r'오후\s*(\d{1,2})시', text)
    if m:
        h = int(m.group(1))
        if h != 12: h += 12
        return f"{h:02d}:00"
    m = re.search(r'오전\s*(\d{1,2})시', text)
    if m:
        h = int(m.group(1))
        if h == 12: h = 0
        return f"{h:02d}:00"
    m = re.search(r'(\d{1,2})시\s*반', text)
    if m: return f"{int(m.group(1)):02d}:30"
    m = re.search(r'(\d{1,2})시', text)
    if m: return f"{int(m.group(1)):02d}:00"
    for k, v in PERIOD_TO_TIME.items():
        if k in text: return v
    return None


def normalize_date(text, base_date=DEFAULT_BASE_DATE):
    text = text.strip()
    m = re.search(r'(\d{1,2})월\s*(\d{1,2})일', text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            d = datetime(base_date.year, month, day)
            if d < base_date - timedelta(days=90):
                d = datetime(base_date.year + 1, month, day)
            return d.strftime('%Y-%m-%d')
        except ValueError:
            return text

    if '이번 주말' in text or '토요일' in text:
        days_until = (5 - base_date.weekday()) % 7
        if days_until == 0: days_until = 7
        target = base_date + timedelta(days=days_until)
        return target.strftime('%Y-%m-%d')
    if '다음 주말' in text:
        days_until = (5 - base_date.weekday()) % 7 + 7
        target = base_date + timedelta(days=days_until)
        return target.strftime('%Y-%m-%d')
    if '내일' in text:
        return (base_date + timedelta(days=1)).strftime('%Y-%m-%d')
    if re.search(r'모레|내일모레', text):
        return (base_date + timedelta(days=2)).strftime('%Y-%m-%d')
    return text


def normalize_duration(text):
    text = text.strip()
    m = re.search(r'(\d+)박\s*(\d+)일', text)
    if m:
        return {'nights': int(m.group(1)), 'days': int(m.group(2)), 'display': f"{m.group(1)}박 {m.group(2)}일"}
    if '당일' in text:
        return {'nights': 0, 'days': 1, 'display': '당일치기'}
    return {'display': text}


def detect_day_keyword(text, current_day):
    for pattern, day_val in DAY_PATTERNS:
        if re.search(pattern, text):
            if day_val == 'NEXT': return current_day + 1
            if day_val in ('WEEKEND', 'NEXT_WEEKEND', 'TOMORROW', 'DAY_AFTER_TOMORROW'):
                return current_day
            return day_val
    return None


def detect_message_time(text, entities):
    for ent in entities:
        if ent['type'] == 'TIME':
            t = normalize_time(ent['text'])
            if t: return t
    for k, v in PERIOD_TO_TIME.items():
        if k in text: return v
    return None


def time_to_minutes(time_str):
    if not time_str: return 9999
    try:
        h, m = time_str.split(':')
        return int(h) * 60 + int(m)
    except:
        return 9999


def is_sub_location(text):
    return any(kw in text for kw in SUB_LOC_KEYWORDS)


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
    if msg_idx < 0 or msg_idx >= len(in_span_msgs): return False
    text = in_span_msgs[msg_idx]['text']
    loc_escaped = _re.escape(loc_text)
    has_recommend = any(_re.search(p.format(loc=loc_escaped), text) for p in LODGING_LOC_PATTERNS)
    if not has_recommend: return False
    start = max(0, msg_idx - lookback)
    for j in range(start, msg_idx):
        prev_text = in_span_msgs[j]['text']
        prev_ents = in_span_msgs[j].get('entities', [])
        if any(kw in prev_text for kw in LODGING_CONTEXT_KEYWORDS) or any(e['type'] == 'LODGING' for e in prev_ents):
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
            if ent['type'] != 'LOC': continue
            role = detect_loc_role(text, ent['text'])
            if role == 'departure':
                if departure_loc is None: departure_loc = ent['text']
                departure_msg_indices.append(i)
            elif role == 'return':
                if ent['text'] not in return_locs: return_locs.append(ent['text'])
    if departure_loc:
        for i, msg in enumerate(in_span_msgs):
            if departure_loc in msg['text']:
                for ent in msg.get('entities', []):
                    if ent['type'] == 'TIME':
                        t = normalize_time(ent['text'])
                        if t: departure_time = t; break
                if departure_time: break
        if not departure_time and departure_msg_indices:
            first = departure_msg_indices[0]
            for j in range(max(0, first-5), min(len(in_span_msgs), first+6)):
                for ent in in_span_msgs[j].get('entities', []):
                    if ent['type'] == 'TIME':
                        t = normalize_time(ent['text'])
                        if t: departure_time = t; break
                if departure_time: break
    departure = {'location': departure_loc, 'time': departure_time} if departure_loc else None
    return departure, return_locs


# ============================================================
# assemble_events
# ============================================================

def assemble_events(messages, lookback=10):
    proposals = []
    current_day = 1
    open_proposals = []
    in_span_msgs = [m for m in messages if m.get('in_travel_span')]
    msg_to_span_idx = {id(m): i for i, m in enumerate(in_span_msgs)}

    for i, msg in enumerate(messages):
        if not msg.get('in_travel_span'): continue
        intent = msg.get('intent')
        ents = msg.get('entities', [])
        text = msg['text']

        day_kw = detect_day_keyword(text, current_day)
        if day_kw is not None:
            if day_kw == -1: current_day = 'LAST'
            elif day_kw == 'NEXT': current_day += 1
            else: current_day = day_kw
            if intent in ('AGREE', 'CONFIRM'):
                for p in reversed(proposals):
                    if (i - p['idx']) > lookback: break
                    if p['status'] in ('pending', 'confirmed'):
                        p['day_hint'] = current_day
                        break

        time_str = detect_message_time(text, ents)

        span_idx = msg_to_span_idx.get(id(msg), -1)
        filtered_ents = [ent for ent in ents if not (ent['type'] == 'LOC' and span_idx >= 0 and is_lodging_context_loc(span_idx, in_span_msgs, ent['text']))]
        ents = filtered_ents

        if intent == 'PROPOSE' and ents:
            has_agree = any(tok in text for tok in AGREE_TOKENS)
            new_proposal = {'idx': i, 'msg': msg, 'entities': ents, 'status': 'pending',
                            'day_hint': current_day, 'time': time_str, 'response_text': None}
            proposals.append(new_proposal)
            open_proposals.append(new_proposal)
            if has_agree:
                new_proposal['status'] = 'confirmed'
                new_proposal['response_text'] = '(self-agree)'
                absorb_general_by_specific(new_proposal, proposals, lookback)
                for p in open_proposals:
                    if p['status'] == 'pending': p['status'] = 'confirmed'
                open_proposals.clear()

        elif intent in ('AGREE', 'CONFIRM'):
            for p in open_proposals:
                if p['status'] == 'pending':
                    p['status'] = 'confirmed'
                    p['response_text'] = text
            open_proposals.clear()

            non_meta = [e for e in ents if e['type'] not in META_TYPES]
            new_meta = [e for e in ents if e['type'] in META_TYPES]
            has_agree = any(tok in text for tok in AGREE_TOKENS)

            if non_meta and has_agree:
                new_p = {'idx': i, 'msg': msg, 'entities': non_meta, 'status': 'confirmed',
                         'day_hint': current_day, 'time': time_str, 'response_text': text}
                proposals.append(new_p)
                absorb_general_by_specific(new_p, proposals, lookback)

            if new_meta:
                proposals.append({'idx': i, 'msg': msg, 'entities': new_meta, 'status': 'confirmed',
                                  'day_hint': current_day, 'time': time_str, 'response_text': None})

        elif intent in ('DISAGREE', 'CANCEL'):
            for p in reversed(proposals):
                if p['status'] in ('pending', 'confirmed') and (i - p['idx']) <= lookback:
                    p['status'] = 'cancelled'
                    p['response_text'] = text
                    break
    return proposals


def absorb_general_by_specific(current, proposals, lookback):
    cur_idx = current['idx']
    cur_texts = [e['text'] for e in current['entities']]
    for p in reversed(proposals):
        if p is current or p['status'] not in ('pending', 'confirmed'): continue
        if cur_idx - p['idx'] > lookback: break
        keep = []
        for ent in p['entities']:
            if ent['type'] in META_TYPES:
                keep.append(ent)
                continue
            same_type = [e['text'] for e in current['entities']
                         if e['type'] == ent['type'] or
                            (ent['type'] == 'ACTIVITY' and e['type'] == 'LOC') or
                            (ent['type'] == 'LOC' and e['type'] == 'ACTIVITY')]
            if same_type and is_subsumed_by(ent['text'], same_type):
                continue
            keep.append(ent)
        p['entities'] = keep


def extract_meta(proposals, all_messages=None):
    meta = {'destination': None, 'duration': None, 'start_date': None,
            'main_lodging': None, 'main_transport': None}

    city_score = defaultdict(float)
    first_confirmed_city = None

    for m in (all_messages or []):
        if not m.get('in_travel_span'): continue
        intent = m.get('intent', '')
        weight = 4.0 if intent in ('AGREE', 'CONFIRM') else 2.0 if intent == 'PROPOSE' else 1.0
        for ent in m.get('entities', []):
            if ent['type'] != 'LOC': continue
            loc = ent['text']
            if not is_true_city_location(loc): continue
            city_score[loc] += weight
            if intent in ('AGREE', 'CONFIRM') and not first_confirmed_city:
                first_confirmed_city = loc

    departure, _ = extract_departure_return(all_messages or [])
    if departure and departure['location'] in city_score:
        city_score[departure['location']] -= 30.0
    if first_confirmed_city and first_confirmed_city in city_score:
        city_score[first_confirmed_city] += 15.0

    if city_score:
        meta['destination'] = max(city_score, key=city_score.get)

    for p in proposals:
        if p['status'] not in ('confirmed', 'pending'): continue
        for e in p['entities']:
            if e['type'] == 'DURATION':
                norm = normalize_duration(e['text'])
                if '박' in e['text'] or '일' in e['text']:
                    meta['duration'] = norm.get('display', e['text'])
                    break
        if meta['duration']: break

    for p in proposals:
        if p['status'] not in ('confirmed', 'pending'): continue
        for e in p['entities']:
            if e['type'] == 'DATE':
                converted = normalize_date(e['text'])
                if converted != e['text']:
                    meta['start_date'] = converted
                    break
        if meta['start_date']: break

    for p in reversed([p for p in proposals if p['status'] == 'confirmed']):
        for e in p['entities']:
            if e['type'] == 'LODGING' and e['text'] not in ('숙소',):
                meta['main_lodging'] = e['text']
                break
        if meta['main_lodging']: break

    for p in [p for p in proposals if p['status'] == 'confirmed']:
        for e in p['entities']:
            if e['type'] == 'TRANSPORT':
                meta['main_transport'] = e['text']
                break
        if meta['main_transport']: break

    return meta


# ============================================================
# v4.2 핵심: build_schedule - 교차 인텐트 병합 강화
# ============================================================

def is_meta_only(entities):
    return all(e['type'] in META_TYPES for e in entities)


def get_total_days(duration_str):
    if not duration_str: return None
    m = re.search(r'(\d+)일', duration_str)
    if m: return int(m.group(1))
    if '당일' in duration_str: return 1
    return None


def build_schedule(proposals, total_days, destination=None, lodging=None, main_transport=None):
    GENERIC_ACTS_FOR_MERGE = {'점심', '저녁', '아침', '식사', '카페', '회', '밥'}
    SEPARATION_WORDS = ['그리고', '그담', '그담에', '먹고', '하고', '갔다가', '그리고나서']

    confirmed_sorted = sorted([p for p in proposals if p['status'] == 'confirmed'], key=lambda p: p['idx'])
    merged_prev_idx = set()
    merge_into_next = {}

    # v4.2 핵심 강화: generic activity + specific place 병합
    for i in range(len(confirmed_sorted)-1):
        prev = confirmed_sorted[i]
        if prev['idx'] in merged_prev_idx: continue

        prev_has_place = any(CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING') for e in prev['entities'])
        if prev_has_place: continue

        prev_generic = [e for e in prev['entities']
                        if CATEGORY_MAP.get(e['type']) in ('ACTIVITY', 'FOOD')
                        and (e['text'] in GENERIC_ACTS_FOR_MERGE or len(e['text']) <= 4)]

        if not prev_generic: continue

        # 다음 3턴 이내에 specific place/FOOD가 있는지 찾기
        nxt = None
        for j in range(i+1, min(i+5, len(confirmed_sorted))):
            cand = confirmed_sorted[j]
            if cand['idx'] - prev['idx'] > 4 or cand.get('day_hint') != prev.get('day_hint'): break
            if any(sw in cand['msg']['text'] for sw in SEPARATION_WORDS): break

            has_specific = any(CATEGORY_MAP.get(e['type']) in ('PLACE', 'LODGING', 'FOOD')
                               and e['text'] not in GENERIC_ACTS_FOR_MERGE
                               and e['text'] not in (destination, lodging, main_transport)
                               for e in cand['entities'])
            if has_specific:
                nxt = cand
                break

        if nxt:
            merged_prev_idx.add(prev['idx'])
            merge_into_next[nxt['idx']] = prev_generic

    # 병합 적용
    for p in proposals:
        if p['idx'] in merged_prev_idx:
            p['status'] = 'absorbed'
    for p in proposals:
        if p['idx'] in merge_into_next:
            p['entities'] = list(p['entities']) + merge_into_next[p['idx']]

    # 본 처리
    confirmed = [p for p in proposals if p['status'] == 'confirmed' and not is_meta_only(p['entities'])]
    GENERIC_NOUNS = {'숙소', '점심', '저녁', '아침', '카페', '식사', '시장'}
    dedup_set = {destination, lodging, main_transport}
    dedup_set.discard(None)

    days_data = defaultdict(list)
    day_groups = defaultdict(list)
    for p in confirmed:
        dh = p.get('day_hint')
        if dh == 'LAST': dh = total_days or 999
        day_groups[dh].append(p)

    inferred_day = 1
    for hint, group in sorted(day_groups.items(), key=lambda x: (isinstance(x[0], int), x[0])):
        group.sort(key=lambda p: time_to_minutes(p.get('time')))
        day = hint if isinstance(hint, int) else inferred_day
        if day == 999 and total_days: day = total_days

        for p in group:
            valid_ents = []
            for ent in p['entities']:
                cat = CATEGORY_MAP.get(ent['type'])
                if not cat: continue
                txt = ent['text']
                if txt in GENERIC_NOUNS and ent['type'] in ('LOC', 'LODGING'): continue
                if txt in dedup_set: continue
                if ent['type'] == 'LOC' and (is_address(txt) or detect_loc_role(p['msg']['text'], txt) != 'event'): continue
                valid_ents.append((ent, cat))
            if not valid_ents: continue

            msg_text = p['msg']['text']
            is_multi = bool(re.search(r'갈거면.*도\s', msg_text)) or bool(re.search(r'도\s+가자', msg_text))
            place_ents = [(e,c) for e,c in valid_ents if c in ('PLACE','LODGING')]
            food_ents = [(e,c) for e,c in valid_ents if c == 'FOOD']
            activity_ents = [(e,c) for e,c in valid_ents if c == 'ACTIVITY']
            memo_parts = [f"{e['text']} 먹기" for e,_ in food_ents] + [e['text'] for e,_ in activity_ents]
            memo = ', '.join(memo_parts) if memo_parts else None

            events_to_add = []
            if place_ents:
                for ent, cat in place_ents:
                    ev = {'time': p['time'], 'location': ent['text'], 'category': cat, 'source_text': p['msg']['text']}
                    if memo: ev['memo'] = memo
                    events_to_add.append(ev)
            elif food_ents:
                for ent, cat in food_ents:
                    ev = {'time': p['time'], 'location': ent['text'], 'category': cat, 'source_text': p['msg']['text']}
                    if memo: ev['memo'] = memo
                    events_to_add.append(ev)
            else:
                for ent, cat in valid_ents:
                    loc_text = ent['text']
                    if cat == 'ACTIVITY' and loc_text in ('점심','저녁','아침','식사'):
                        loc_text = f"{loc_text} 식사" if loc_text != '식사' else '식사'
                    ev = {'time': p['time'], 'location': loc_text, 'category': cat, 'source_text': p['msg']['text']}
                    events_to_add.append(ev)

            for idx, ev in enumerate(events_to_add):
                if not ev.get('time'):
                    slot = (len(days_data[day]) + idx) % 4 + 1
                    ev['time'] = DEFAULT_TIME_SLOTS.get(slot, '12:00')
                days_data[day].append(ev)

        inferred_day = day + 1

    if 999 in days_data and total_days:
        days_data[total_days].extend(days_data.pop(999))

    sorted_days = sorted(days_data.keys())
    days_list = []
    for d_idx, day in enumerate(sorted_days, 1):
        events = days_data[day]
        events.sort(key=lambda e: time_to_minutes(e['time']))
        seen = set()
        unique = []
        for e in events:
            key = (e['location'], e['category'])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        days_list.append({'day': d_idx, 'events': unique})
    return days_list


def build_pending_cancelled(proposals, dedup_set):
    pending = []
    cancelled = []
    GENERIC_NOUNS = {'숙소', '점심', '저녁', '아침', '카페', '식사', '시장'}
    for p in proposals:
        for ent in p['entities']:
            if ent['type'] in META_TYPES: continue
            cat = CATEGORY_MAP.get(ent['type'])
            if not cat: continue
            txt = ent['text']
            if txt in GENERIC_NOUNS or txt in dedup_set: continue
            if ent['type'] == 'LOC' and (is_address(txt) or detect_loc_role(p['msg']['text'], txt) != 'event'): continue
            item = {'category': cat, 'location': txt, 'source_text': p['msg']['text']}
            if p['status'] == 'pending':
                pending.append(item)
            elif p['status'] == 'cancelled':
                item['cancel_reason'] = p.get('response_text', '')
                cancelled.append(item)
    return pending, cancelled


# ============================================================
# 메인
# ============================================================

def split_into_chats(results):
    if results and results[0].get('source'):
        chats = defaultdict(list)
        for r in results: chats[r['source']].append(r)
        return list(chats.items())
    return [('chat_1', results)]


def process(pred_results, verbose=False):
    chats = split_into_chats(pred_results)
    summaries = []
    for chat_id, chat_msgs in chats:
        if verbose: print(f"\n[{chat_id}] 처리 중...")
        proposals = assemble_events(chat_msgs)
        meta = extract_meta(proposals, all_messages=chat_msgs)
        departure, return_via = extract_departure_return(chat_msgs)
        total_days = get_total_days(meta['duration'])
        dedup_set = {meta['destination'], meta['main_lodging'], meta['main_transport']}
        dedup_set.discard(None)
        days = build_schedule(proposals, total_days, meta['destination'], meta['main_lodging'], meta['main_transport'])
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
            print(f"  목적지: {meta['destination']}, 기간: {meta['duration']}, 시작: {meta['start_date']}")
        summaries.append(summary)
    return summaries


def format_human_readable(summary):
    lines = ["\n" + "="*60, f"여행 일정표 [{summary['chat_id']}] (v4.2)", "="*60]
    if summary['destination']: lines.append(f"📍 목적지   : {summary['destination']}")
    if summary['duration']: lines.append(f"📅 기간     : {summary['duration']}")
    if summary['start_date']: lines.append(f"🗓  날짜     : {summary['start_date']}")
    if summary['transport']: lines.append(f"🚗 이동     : {summary['transport']}")
    if summary['lodging']: lines.append(f"🏨 숙소     : {summary['lodging']}")
    if summary.get('departure'):
        dep = summary['departure']
        t = f" ({dep['time']})" if dep.get('time') else ""
        lines.append(f"🚉 출발지   : {dep['location']}{t}")
    if summary.get('return_via'):
        lines.append(f"↩️  귀환 경유 : {', '.join(summary['return_via'])}")
    lines.append("")
    for day in summary['days']:
        lines.append(f"━━━ Day {day['day']} ━━━")
        for ev in day['events']:
            t = ev['time'] if ev['time'] else '  -  '
            cat = ev['category']
            loc = ev['location']
            memo = ev.get('memo')
            line = f"  {t:>5s}  [{cat:8s}] {loc}"
            if memo: line += f"  ({memo})"
            lines.append(line)
        lines.append("")
    if summary['pending']:
        lines.append(f"⏳ 미확정 ({len(summary['pending'])}건)")
        for p in summary['pending'][:5]: lines.append(f"   - [{p['category']}] {p['location']}")
    if summary['cancelled']:
        lines.append(f"❌ 취소됨 ({len(summary['cancelled'])}건)")
        for c in summary['cancelled'][:5]:
            reason = (c.get('cancel_reason','') or '')[:25]
            lines.append(f"   - [{c['category']}] {c['location']}  ({reason})")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--output', default='summary.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"v4.2 최종 rule-based 완성판 실행 → 입력: {args.pred}")
    with open(args.pred, encoding='utf-8') as f:
        pred = json.load(f)
    summaries = process(pred, verbose=args.verbose)
    print(f"\nv4.2 처리 완료\n")
    for s in summaries:
        print(format_human_readable(s))
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"✓ summary.json 저장 완료")
