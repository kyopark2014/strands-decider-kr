#!/usr/bin/env python3
"""strands-decider System One API 예제 테스트.

테스트 케이스는 testcases.json 의 cases 를 사용합니다.

로컬 서버가 떠 있어야 합니다:
  strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v2-2611 --port 8000

실행:
  python test_systemone_examples.py
  python test_systemone_examples.py --base-url http://localhost:8000 --only 1,4
  python test_systemone_examples.py --cases ./testcases.json --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = ROOT / "testcases.json"


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    """testcases.json 에서 cases 목록을 로드합니다."""
    cases_path = path or DEFAULT_CASES_PATH
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases")
    else:
        raise ValueError(f"지원하지 않는 testcases 형식: {cases_path}")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"cases 가 비어 있거나 배열이 아닙니다: {cases_path}")
    return cases


# run_strands_devider.py 등이 import 할 수 있도록 모듈 로드 시 기본 케이스 준비
EXAMPLES: list[dict[str, Any]] = load_examples()


def post_systemone(base_url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/systemone"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def format_criteria(question: dict[str, Any]) -> list[str]:
    """instructions와 함께 눈으로 검증하기 쉽게 criteria를 줄 단위로 포맷."""
    kind = question.get("type")
    criteria = question.get("criteria")
    lines: list[str] = []

    if kind == "noul":
        if isinstance(criteria, dict) and criteria:
            for key in ("true", "false"):
                if key in criteria and criteria[key] is not None:
                    lines.append(f"      {key}: {criteria[key]}")
            for key, value in criteria.items():
                if key not in ("true", "false"):
                    lines.append(f"      {key}: {value}")
        else:
            lines.append("      true=예 / false=아니오  (값은 P(true), 1에 가까울수록 예)")
        return lines

    if kind == "choice" and isinstance(criteria, dict):
        for key, value in criteria.items():
            if value is None:
                lines.append(f"      - {key}")
            else:
                lines.append(f"      - {key}: {value}")
        return lines

    if kind == "score" and isinstance(criteria, list):
        for i, label in enumerate(criteria):
            lines.append(f"      {i}: {label}")
        return lines

    if criteria is not None:
        lines.append(f"      {json.dumps(criteria, ensure_ascii=False)}")
    else:
        lines.append("      (없음)")
    return lines


def format_qa(name: str, question: dict[str, Any], answer: dict[str, Any] | None) -> str:
    """질문(instructions/criteria)과 결과를 한 블록으로 묶어 검증하기 쉽게 출력."""
    kind = question.get("type")
    lines = [
        f"  ---- {name} ({kind}) ----",
        f"  instructions: {question.get('instructions')}",
        "  criteria:",
        *format_criteria(question),
    ]

    if answer is None:
        lines.append("  result: (요청 실패)")
        return "\n".join(lines)

    if kind == "noul":
        value = answer["noul"]
        leaning = "예(true) 쪽" if value >= 0.5 else "아니오(false) 쪽"
        lines.append(f"  result: noul={value:.4f}  → {leaning}")
    elif kind == "choice":
        probs = answer.get("probabilities", {})
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        lines.append(
            f"  result: choice={answer['choice']}  "
            f"(confidence={answer.get('confidence', 0):.3f})"
        )
        lines.append("  probabilities:")
        for opt, prob in ranked:
            mark = " ← 선택" if opt == answer.get("choice") else ""
            lines.append(f"      {opt}: {prob:.3f}{mark}")
    elif kind == "score":
        legend = answer.get("legend", {})
        score = answer["score"]
        # 가장 가까운 레벨 라벨로 해석 힌트
        nearest = min(legend.items(), key=lambda kv: abs(int(kv[0]) - score)) if legend else None
        hint = f"  ≈ {nearest[1]}" if nearest else ""
        lines.append(
            f"  result: score={score:.4f}{hint}  "
            f"(confidence={answer.get('confidence', 0):.3f})"
        )
        probs = answer.get("probabilities", {})
        if probs:
            lines.append("  probabilities:")
            for idx, label in legend.items():
                prob = probs.get(str(idx), probs.get(idx, 0.0))
                lines.append(f"      {idx}:{label} = {prob:.3f}")
    else:
        lines.append(f"  result: {json.dumps(answer, ensure_ascii=False)}")

    return "\n".join(lines)


def run_examples(
    base_url: str,
    selected: list[int] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> int:
    cases = examples if examples is not None else EXAMPLES
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"서버에 연결할 수 없습니다: {health_url}")
        print(f"  {exc}")
        print("먼저 서버를 실행하세요:")
        print(
            "  strands-decider serve "
            "StrandsAgents/strands-decider-E4B-gemma4-v2-2611 --port 8000"
        )
        return 1

    print(f"health: model={health.get('model')} device={health.get('device')} status={health.get('status')}")
    print(f"cases: {len(cases)}")
    print("=" * 72)

    failures = 0
    timings_ms: list[float] = []
    total_started = time.perf_counter()

    for idx, example in enumerate(cases, start=1):
        if selected is not None and idx not in selected:
            continue

        request = example["request"]
        questions = request.get("questions", {})

        print(f"\n[{idx:02d}] {example['name']}")
        print(f"     {example['description']}")
        print(f"  state: {format_state(request.get('state'))}")

        started = time.perf_counter()
        try:
            result = post_systemone(base_url, request)
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            body = exc.read().decode("utf-8", errors="replace")
            print(f"  ERROR HTTP {exc.code}: {body}")
            for qname, question in questions.items():
                print(format_qa(qname, question, None))
            print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001 - demo script
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            print(f"  ERROR: {exc}")
            for qname, question in questions.items():
                print(format_qa(qname, question, None))
            print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")
            failures += 1
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)

        answers = result.get("answers", {})
        for qname, question in questions.items():
            print(format_qa(qname, question, answers.get(qname)))

        usage = result.get("usage", {})
        server_ms = result.get("latency_ms")
        print(
            f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)"
            + (f" / {server_ms} ms (서버)" if server_ms is not None else "")
        )
        print(
            f"  model={result.get('model')} "
            f"tokens={usage.get('input_tokens')}/{usage.get('output_tokens')}"
        )

    total_ms = (time.perf_counter() - total_started) * 1000
    print("\n" + "=" * 72)
    if timings_ms:
        avg_ms = sum(timings_ms) / len(timings_ms)
        print(
            f"실행시간 요약: 총 {total_ms:.2f} ms | "
            f"예제 {len(timings_ms)}개 | "
            f"평균 {avg_ms:.2f} ms | "
            f"최소 {min(timings_ms):.2f} ms | "
            f"최대 {max(timings_ms):.2f} ms"
        )
    if failures:
        print(f"완료: 실패 {failures}건")
        return 1
    print("완료: 모든 예제 성공")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strands-decider /v1/systemone 예제 테스트")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"서버 URL (기본: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"테스트 케이스 JSON (기본: {DEFAULT_CASES_PATH})",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="실행할 예제 번호 (쉼표 구분, 예: 1,4,8)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="예제 목록만 출력",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        examples = load_examples(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"testcases 로드 실패: {exc}")
        return 1

    if args.list:
        print(f"cases file: {args.cases}")
        for idx, example in enumerate(examples, start=1):
            print(f"{idx:02d}  {example['name']}  —  {example['description']}")
        return 0

    selected = None
    if args.only:
        selected = [int(x.strip()) for x in args.only.split(",") if x.strip()]

    return run_examples(args.base_url, selected, examples=examples)


if __name__ == "__main__":
    sys.exit(main())
