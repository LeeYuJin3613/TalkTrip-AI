"""
TalkTrip Hybrid Schedule Builder v1.5.3 Final
Rule-based + Local LLM (Refine)
"""

import json
import inspect
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# ==================== Rule-based Import ====================
RULE_BASED_AVAILABLE = False
_rule_based_process: Optional[Callable] = None
_rule_based_supports_verbose = False

try:
    from .schedule_builder import process as _rb_func
    _rule_based_process = _rb_func
    RULE_BASED_AVAILABLE = True
except ImportError:
    try:
        from .schedule_builder import build_schedule as _rb_func
        _rule_based_process = _rb_func
        RULE_BASED_AVAILABLE = True
    except ImportError:
        logger.error("Rule-based schedule builder를 import할 수 없습니다.")

if _rule_based_process is not None:
    try:
        sig = inspect.signature(_rule_based_process)
        _rule_based_supports_verbose = "verbose" in sig.parameters
    except (ValueError, TypeError):
        _rule_based_supports_verbose = False


def _call_rule_based(analyzed_messages: List[Dict[str, Any]], verbose: bool = True) -> Any:
    if _rule_based_process is None:
        raise RuntimeError("Rule-based Schedule Builder를 사용할 수 없습니다.")
    if _rule_based_supports_verbose:
        return _rule_based_process(analyzed_messages, verbose=verbose)
    return _rule_based_process(analyzed_messages)


# ==================== LLM Configuration ====================
LLM_MODEL_NAME = "PUT_YOUR_MODEL_HERE"   # ← 여기에 원하는 모델명 입력
# 예시:
# "gemma4", "gemma2:9b", "qwen2.5:14b", "llama3.1:8b", "gemma4:27b" 등

_LLM_INIT_ATTEMPTED = False
_chain: Optional[Any] = None
MAX_LLM_INPUT_CHARS = 12000


def _get_chain():
    """LLM chain lazy initialization"""
    global _LLM_INIT_ATTEMPTED, _chain
    if _LLM_INIT_ATTEMPTED:
        return _chain
    _LLM_INIT_ATTEMPTED = True

    if LLM_MODEL_NAME == "PUT_YOUR_MODEL_HERE":
        logger.warning("LLM 모델이 설정되지 않았습니다. Rule-based만 사용합니다.")
        return None

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        llm = ChatOllama(
            model=LLM_MODEL_NAME,
            temperature=0.2,
            num_ctx=8192,
            format="json"
        )

        SYSTEM_PROMPT = """당신은 여행 일정 전문 AI 어시스턴트입니다.
Rule-based 초안을 기반으로 더 자연스럽고 논리적인 일정을 만들어주세요.

[중요]
1. 핵심 구조, 날짜, 시간, 장소는 최대한 유지
2. 비효율적인 동선은 개선
3. **동일한 JSON 구조로만 출력** (설명 절대 금지)"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """
=== 사용자 원본 메시지 ===
{raw_messages}

=== Stage 분석 결과 ===
{stage_results}

=== Rule-based 초안 ===
{rule_based}

위 정보를 바탕으로 최적화된 최종 일정을 JSON으로 출력해주세요.
""")
        ])

        _chain = prompt | llm | JsonOutputParser()
        logger.info(f"✅ LLM 초기화 완료 (Model: {LLM_MODEL_NAME})")

    except ImportError:
        logger.warning("langchain_ollama 미설치 → Rule-based만 사용")
    except Exception as e:
        logger.error(f"LLM 초기화 실패: {e}")

    return _chain


# ==================== Helpers ====================
def _safe_json_dumps(obj: Any, **kwargs) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", str)
    return json.dumps(obj, **kwargs)


def _validate_refined(refined: Any, original: Any) -> bool:
    if isinstance(original, dict) and isinstance(refined, dict):
        missing = set(original.keys()) - set(refined.keys())
        if missing:
            logger.warning(f"LLM 결과 누락 키: {missing}")
            return False
        return True
    if isinstance(original, list):
        return isinstance(refined, list) and len(refined) > 0
    return type(refined) is type(original)


# ==================== Main Function ====================
def build_schedule_hybrid(
    analyzed_messages: List[Dict[str, Any]],
    verbose: bool = True,
) -> Any:
    if not RULE_BASED_AVAILABLE:
        raise RuntimeError("Rule-based Schedule Builder를 사용할 수 없습니다.")

    rule_result = _call_rule_based(analyzed_messages, verbose=verbose)

    chain = _get_chain()
    if chain is None:
        if verbose:
            logger.info("LLM을 사용할 수 없어 Rule-based 결과만 반환합니다.")
        return rule_result

    raw_texts = [
        m.get("text", m.get("message", ""))
        for m in analyzed_messages
        if isinstance(m, dict)
    ]

    if not raw_texts:
        logger.warning("원본 메시지가 비어있어 LLM Refine을 건너뜁니다.")
        return rule_result

    llm_input = {
        "rule_based": _safe_json_dumps(rule_result, indent=2),
        "stage_results": _safe_json_dumps(analyzed_messages, indent=2),
        "raw_messages": "\n".join(raw_texts),
    }

    if sum(len(v) for v in llm_input.values()) > MAX_LLM_INPUT_CHARS:
        logger.warning("LLM 입력이 너무 깁니다. Rule-based 결과만 반환.")
        return rule_result

    try:
        if verbose:
            logger.info("LLM Refine 시작...")
        refined_result = chain.invoke(llm_input)

        if not _validate_refined(refined_result, rule_result):
            logger.warning("LLM 결과 스키마 검증 실패 → Rule-based 반환")
            return rule_result

        if verbose:
            logger.info("✅ LLM Refine 완료")
        return refined_result

    except Exception as e:
        logger.error(f"LLM Refine 실패: {e}")
        return rule_result


def process(analyzed_messages: List[Dict[str, Any]], verbose: bool = True) -> Any:
    return build_schedule_hybrid(analyzed_messages, verbose=verbose)
