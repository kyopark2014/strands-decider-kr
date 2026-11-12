#!/usr/bin/env python3
"""SageMaker Real-time Endpoint 설치 스크립트.

deploy/sagemaker CDK로 strands-decider 모델을 배포하고,
결과를 config.json에 저장합니다.

기본값 (config에 없으면):
  model.hub_id      = StrandsAgents/strands-decider-E4B-gemma4-v1-2610
  instance_type     = ml.g5.xlarge
  region            = AWS 세션 기본 리전 (없으면 us-west-2)

실행:
  python installer.py
  python installer.py --instance-type ml.g5.xlarge
  python installer.py --model StrandsAgents/strands-decider-E4B-gemma4-v1-2610
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
CDK_DIR = SCRIPT_DIR / "deploy" / "sagemaker"

DEFAULT_MODEL = "StrandsAgents/strands-decider-E4B-gemma4-v1-2610"
DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_REGION = "us-west-2"
DEFAULT_REVISION = "main"
DEFAULT_DLC_TAG = "serve-sagemaker-cuda-v1"


def log(msg: str) -> None:
    print(msg, flush=True)


def endpoint_name_for(model: str) -> str:
    """deploy/sagemaker/lib/decider-endpoint-stack.ts 와 동일한 규칙."""
    name = (model.split("/")[-1] if "/" in model else model).lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name).strip("-")
    if not re.fullmatch(r"[a-z0-9](-*[a-z0-9]){0,62}", name):
        raise ValueError(f"엔드포인트 이름을 만들 수 없습니다: {model}")
    return name


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            log(f"config.json 파싱 실패, 새로 작성합니다: {exc}")
    return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def resolve_identity(region: str) -> tuple[str, str]:
    session = boto3.Session(region_name=region)
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    return identity["Account"], identity["Arn"]


def ensure_defaults(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    session = boto3.Session()
    region = (
        args.region
        or config.get("region")
        or session.region_name
        or DEFAULT_REGION
    )
    account_id, _ = resolve_identity(region)

    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    sm = config.get("sagemaker") if isinstance(config.get("sagemaker"), dict) else {}
    cdk = sm.get("cdk_context") if isinstance(sm.get("cdk_context"), dict) else {}

    hub_id = args.model or model.get("hub_id") or cdk.get("model") or DEFAULT_MODEL
    instance_type = (
        args.instance_type
        or sm.get("instance_type")
        or cdk.get("instance_type")
        or DEFAULT_INSTANCE_TYPE
    )
    instance_count = int(
        args.instance_count
        or sm.get("instance_count")
        or cdk.get("instance_count")
        or 1
    )
    revision = model.get("revision") or cdk.get("model_revision") or DEFAULT_REVISION
    endpoint_name = endpoint_name_for(hub_id)

    deep_update(
        config,
        {
            "project": config.get("project") or "strands-decider-kr",
            "account_id": account_id,
            "region": region,
            "model": {
                "hub_id": hub_id,
                "revision": revision,
                "served_name": hub_id.split("/")[-1],
            },
            "sagemaker": {
                "stack_name": endpoint_name,
                "endpoint_name": endpoint_name,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "deploy_source": "deploy/sagemaker",
                "cdk_context": {
                    "model": hub_id,
                    "model_revision": revision,
                    "instance_type": instance_type,
                    "instance_count": instance_count,
                    "dlc_image_tag": cdk.get("dlc_image_tag") or DEFAULT_DLC_TAG,
                },
            },
            "notes": {
                "destroy": (
                    "python uninstaller.py"
                ),
                "invoke_permission": "sagemaker:InvokeEndpoint",
                "sort_json_keys": True,
            },
        },
    )
    return config


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def ensure_node_deps() -> None:
    if not CDK_DIR.is_dir():
        raise FileNotFoundError(f"CDK 디렉터리가 없습니다: {CDK_DIR}")
    if not (CDK_DIR / "node_modules").is_dir():
        if not shutil.which("npm"):
            raise RuntimeError("npm 이 필요합니다 (Node.js 22+).")
        run(["npm", "install"], cwd=CDK_DIR)


def ensure_cdk_bootstrap(account_id: str, region: str) -> None:
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        resp = cfn.describe_stacks(StackName="CDKToolkit")
        status = resp["Stacks"][0]["StackStatus"]
        if status.endswith("_COMPLETE"):
            log(f"CDK bootstrap 확인: CDKToolkit ({status})")
            return
    except ClientError:
        pass

    log("CDK bootstrap 실행...")
    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = region
    run(
        ["npx", "cdk", "bootstrap", f"aws://{account_id}/{region}"],
        cwd=CDK_DIR,
        env=env,
    )


def cdk_context_args(config: dict[str, Any]) -> list[str]:
    sm = config["sagemaker"]
    cdk = sm["cdk_context"]
    args = [
        "-c", f"model={cdk['model']}",
        "-c", f"model_revision={cdk.get('model_revision', DEFAULT_REVISION)}",
        "-c", f"instance_type={cdk['instance_type']}",
        "-c", f"instance_count={cdk.get('instance_count', 1)}",
    ]
    if cdk.get("dlc_image_tag"):
        args.extend(["-c", f"dlc_image_tag={cdk['dlc_image_tag']}"])
    if cdk.get("fallback_instance_types"):
        args.extend(["-c", f"fallback_instance_types={cdk['fallback_instance_types']}"])
    if cdk.get("max_concurrency") not in (None, ""):
        args.extend(["-c", f"max_concurrency={cdk['max_concurrency']}"])
    if cdk.get("serve_args"):
        args.extend(["-c", f"serve_args={cdk['serve_args']}"])
    return args


def deploy_endpoint(config: dict[str, Any]) -> None:
    region = config["region"]
    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_ACCOUNT"] = str(config["account_id"])

    cmd = [
        "npx",
        "cdk",
        "deploy",
        *cdk_context_args(config),
        "--require-approval",
        "never",
        "--progress",
        "events",
    ]
    run(cmd, cwd=CDK_DIR, env=env)


def refresh_config_from_aws(config: dict[str, Any]) -> dict[str, Any]:
    region = config["region"]
    endpoint_name = config["sagemaker"]["endpoint_name"]
    stack_name = config["sagemaker"]["stack_name"]

    cfn = boto3.client("cloudformation", region_name=region)
    sm = boto3.client("sagemaker", region_name=region)

    stack = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack.get("Outputs", [])}

    endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config_name = endpoint["EndpointConfigName"]
    endpoint_config = sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
    variant = endpoint_config["ProductionVariants"][0]

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    deep_update(
        config,
        {
            "sagemaker": {
                "stack_name": stack_name,
                "stack_arn": stack["StackId"],
                "endpoint_name": endpoint_name,
                "endpoint_arn": endpoint["EndpointArn"],
                "endpoint_config_name": endpoint_config_name,
                "endpoint_status": endpoint["EndpointStatus"],
                "instance_type": variant.get("InstanceType"),
                "instance_count": int(variant.get("InitialInstanceCount", 1)),
                "device": outputs.get("Device", config["sagemaker"].get("device", "cuda")),
                "invoke_url": outputs.get(
                    "InvokeUrl",
                    f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
                ),
                "image_uri": outputs.get("ImageUri"),
                "model_data_s3_uri": outputs.get("ModelDataS3Uri"),
                "deployed_at": now,
                "deploy_source": "deploy/sagemaker",
            }
        },
    )
    return config


def smoke_test(config: dict[str, Any]) -> dict[str, Any] | None:
    region = config["region"]
    endpoint_name = config["sagemaker"]["endpoint_name"]
    sort_keys = bool((config.get("notes") or {}).get("sort_json_keys", True))

    payload = {
        "state": "도와주세요! 정산금 지급이 3일째 계속 실패하고 있어요!",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "이 메시지에 긴급함이 느껴지나요?",
            }
        },
    }
    runtime = boto3.client("sagemaker-runtime", region_name=region)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).encode("utf-8")
    resp = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )
    result = json.loads(resp["Body"].read().decode("utf-8"))
    answer = (result.get("answers") or {}).get("is_urgent") or {}
    smoke = {
        "noul": answer.get("noul"),
        "latency_ms": result.get("latency_ms"),
        "state": payload["state"],
        "question": payload["questions"]["is_urgent"]["instructions"],
        "model": result.get("model"),
    }
    config["smoke_test"] = smoke
    log(
        f"스모크 테스트 OK: noul={smoke.get('noul')}  "
        f"latency_ms={smoke.get('latency_ms')}  model={smoke.get('model')}"
    )
    return smoke


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strands-decider SageMaker endpoint 설치")
    parser.add_argument("--region", default=None, help="AWS 리전 (기본: config 또는 us-west-2)")
    parser.add_argument("--model", default=None, help=f"HF Hub id (기본: {DEFAULT_MODEL})")
    parser.add_argument(
        "--instance-type",
        default=None,
        help=f"SageMaker 인스턴스 (기본: {DEFAULT_INSTANCE_TYPE})",
    )
    parser.add_argument("--instance-count", type=int, default=None, help="인스턴스 수 (기본: 1)")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="배포 후 스모크 테스트 생략",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(SCRIPT_DIR)

    log("=" * 60)
    log("strands-decider-kr — SageMaker Endpoint 설치")
    log("=" * 60)

    try:
        config = ensure_defaults(load_config(), args)
        save_config(config)

        log(f"project     : {config.get('project')}")
        log(f"account_id  : {config.get('account_id')}")
        log(f"region      : {config.get('region')}")
        log(f"model       : {config['model']['hub_id']}")
        log(f"instance    : {config['sagemaker']['instance_type']} x{config['sagemaker']['instance_count']}")
        log(f"endpoint    : {config['sagemaker']['endpoint_name']}")

        if not shutil.which("npx"):
            log("Error: npx(Node.js 22+)가 필요합니다.")
            return 1

        ensure_node_deps()
        ensure_cdk_bootstrap(str(config["account_id"]), config["region"])
        deploy_endpoint(config)
        config = refresh_config_from_aws(config)
        save_config(config)

        if not args.skip_smoke:
            smoke_test(config)
            save_config(config)

    except NoCredentialsError:
        log("Error: AWS 자격 증명을 찾을 수 없습니다.")
        return 1
    except subprocess.CalledProcessError as exc:
        log(f"Error: 명령 실패 (exit {exc.returncode})")
        return exc.returncode or 1
    except (ClientError, OSError, ValueError, RuntimeError) as exc:
        log(f"Error: {exc}")
        return 1

    log("\n" + "=" * 60)
    log("설치 완료")
    log("=" * 60)
    log(f"endpoint : {config['sagemaker'].get('endpoint_name')}")
    log(f"status   : {config['sagemaker'].get('endpoint_status')}")
    log(f"invoke   : {config['sagemaker'].get('invoke_url')}")
    log(f"config   : {CONFIG_PATH}")
    log("테스트    : python run_strands_devider.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
