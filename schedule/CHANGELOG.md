# Changelog

TalkTrip-AI Schedule Builder (Stage 4-6) 변경 이력

## [v4.2] 

- 교차 인텐트 장소 병합 로직 대폭 강화 (가장 큰 개선점)
  - "점심 먹자" + "중앙시장 가자", "회 먹자" + "만석닭강정" 등 자연스러운 병합
  - generic activity + specific place/sub-location 병합 안정화
- subsumption 로직 추가 강화 (식당명 구체화 처리)
- memo 과다 생성 방지 및 정리
- rule-based travel schedule pipeline 

## [v4.1] 

- `CITY_KEYWORDS` 대폭 확장 (40개 이상 주요 여행 도시/지역 추가)
- `SUB_LOC_KEYWORDS` 보강 (실제 대화에서 자주 등장하는 지명 추가)
- Destination 판단 정확도 및 일반화 성능 향상

## [v4] 

- Time 정보가 없는 이벤트에 기본 시간대 임시 부여 (`10:00`, `12:00`, `15:00`, `18:00`)
- Subsumption 로직 대폭 강화 (Levenshtein ratio + 안전장치)
- build_schedule 내부 병합 로직 개선
- memo 처리 안정화

## [v3] 

- Destination scoring 완전 재설계
  - confirmed 우선 + departure 제외 + sub-location 강력 필터링
- `normalize_date()` 대폭 강화 ("이번 주말", "토요일", "이번 주" 등 상대적 표현 처리)
- Day reversal 버그 완전 해결
- `is_true_city_location()` 함수 도입

## [v2] 

- Destination scoring을 가중치 기반으로 변경
- Day 배정 로직 전체 교체 (시간 역전 버그 해결)
- lookback 동적화 + open proposal stack 도입
- normalize 함수들 상대적 표현 지원 확대
- is_subsumed_by()에 Levenshtein ratio 추가

---

**현재 상태**: rule-based 파이프라인은 **v4.2로 가완성**되었습니다.  
이후 작업은 하이브리드 구조(LLM post-processing + 장소 API)로 진행 고민중.
