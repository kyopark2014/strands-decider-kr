#!/usr/bin/env python3
"""SageMaker endpoint로 strands-decider 예제를 실행합니다.

config.json 에서 region / endpoint 정보를 읽고,
testcases.json 의 cases 를 사용해 단독 실행합니다.

실행:
  python run_strands_devider.py
  python run_strands_devider.py --only 1,4,8
  python run_strands_devider.py --list
  python run_strands_devider.py --config ./config.json --cases ./testcases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_CASES = ROOT / "testcases.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config 루트는 object여야 합니다: {path}")
    return data


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"testcases 파일이 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict):
        cases = data.get("cases")
    else:
        raise ValueError(f"지원하지 않는 testcases 형식: {path}")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"cases 가 비어 있거나 배열이 아닙니다: {path}")
    return cases


def format_state(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def format_criteria(question: dict[str, Any]) -> list[str]:
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
        nearest = (
            min(legend.items(), key=lambda kv: abs(int(kv[0]) - score)) if legend else None
        )
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


def invoke_endpoint(
    *,
    runtime: Any,
    endpoint_name: str,
    payload: dict[str, Any],
    sort_keys: bool = True,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).encode("utf-8")
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    raw = response["Body"].read()
    return json.loads(raw.decode("utf-8"))


def run_examples(
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    selected: list[int] | None = None,
    *,
    cases_path: Path | None = None,
) -> int:
    sm = config.get("sagemaker") or {}
    notes = config.get("notes") or {}
    model = config.get("model") or {}

    region = config.get("region") or "us-west-2"
    endpoint_name = sm.get("endpoint_name")
    if not endpoint_name:
        print("config.sagemaker.endpoint_name 이 필요합니다.")
        return 1

    sort_keys = bool(notes.get("sort_json_keys", True))
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    print(
        f"endpoint={endpoint_name}  region={region}  "
        f"instance={sm.get('instance_type')}  device={sm.get('device')}  "
        f"model={model.get('served_name') or model.get('hub_id')}"
    )
    if cases_path is not None:
        print(f"cases={cases_path} ({len(cases)}개)")
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
            result = invoke_endpoint(
                runtime=runtime,
                endpoint_name=endpoint_name,
                payload=request,
                sort_keys=sort_keys,
            )
        except (ClientError, BotoCoreError, json.JSONDecodeError) as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            timings_ms.append(elapsed_ms)
            print(f"  ERROR: {exc}")
            for qname, question in questions.items():
                print(format_qa(qname, question, None))
            print(f"  실행시간: {elapsed_ms:.2f} ms (클라이언트)")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001 - runner script
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
    parser = argparse.ArgumentParser(
        description="config.json + testcases.json → SageMaker endpoint 실행"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"config 경로 (기본: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"testcases 경로 (기본: {DEFAULT_CASES})",
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
        cases = load_cases(args.cases)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"testcases 로드 실패: {exc}")
        return 1

    if args.list:
        print(f"cases file: {args.cases}")
        for idx, example in enumerate(cases, start=1):
            print(f"{idx:02d}  {example['name']}  —  {example['description']}")
        return 0

    selected = None
    if args.only:
        selected = [int(x.strip()) for x in args.only.split(",") if x.strip()]

    try:
        config = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"config 로드 실패: {exc}")
        return 1

    return run_examples(config, cases, selected, cases_path=args.cases)


if __name__ == "__main__":
    sys.exit(main())
