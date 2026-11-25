#!/usr/bin/env python3
"""SageMaker Real-time Endpoint 삭제 스크립트.

deploy/sagemaker CDK 스택을 destroy하고 config.json의
sagemaker 배포 필드를 정리합니다.

실행:
  python uninstaller.py
  python uninstaller.py --yes
  python uninstaller.py --model StrandsAgents/strands-decider-E4B-gemma4-v2-2611
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
CDK_DIR = SCRIPT_DIR / "deploy" / "sagemaker"

DEFAULT_MODEL = "StrandsAgents/strands-decider-E4B-gemma4-v2-2611"
DEFAULT_INSTANCE_TYPE = "ml.g5.xlarge"
DEFAULT_REGION = "us-west-2"
DEFAULT_REVISION = "main"


def log(msg: str) -> None:
    print(msg, flush=True)


def endpoint_name_for(model: str) -> str:
    name = (model.split("/")[-1] if "/" in model else model).lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name).strip("-")
    if not re.fullmatch(r"[a-z0-9](-*[a-z0-9]){0,62}", name):
        raise ValueError(f"엔드포인트 이름을 만들 수 없습니다: {model}")
    return name


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        log(f"config.json 파싱 실패: {exc}")
        return {}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_deploy_params(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[str, str, str, int, str]:
    session = boto3.Session()
    region = (
        args.region
        or config.get("region")
        or session.region_name
        or DEFAULT_REGION
    )
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    sm = config.get("sagemaker") if isinstance(config.get("sagemaker"), dict) else {}
    cdk = sm.get("cdk_context") if isinstance(sm.get("cdk_context"), dict) else {}

    hub_id = args.model or model_cfg.get("hub_id") or cdk.get("model") or DEFAULT_MODEL
    instance_type = (
        args.instance_type
        or sm.get("instance_type")
        or cdk.get("instance_type")
        or DEFAULT_INSTANCE_TYPE
    )
    instance_count = int(
        sm.get("instance_count") or cdk.get("instance_count") or 1
    )
    revision = model_cfg.get("revision") or cdk.get("model_revision") or DEFAULT_REVISION
    return region, hub_id, instance_type, instance_count, revision


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    log(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def stack_exists(region: str, stack_name: str) -> bool:
    cfn = boto3.client("cloudformation", region_name=region)
    try:
        status = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["StackStatus"]
        return status != "DELETE_COMPLETE"
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ValidationError", "ResourceNotFoundException"}:
            return False
        raise


def destroy_stack(
    *,
    region: str,
    account_id: str,
    hub_id: str,
    revision: str,
    instance_type: str,
    instance_count: int,
) -> None:
    if not CDK_DIR.is_dir():
        raise FileNotFoundError(f"CDK 디렉터리가 없습니다: {CDK_DIR}")
    if not shutil.which("npx"):
        raise RuntimeError("npx(Node.js 22+)가 필요합니다.")
    if not (CDK_DIR / "node_modules").is_dir():
        run(["npm", "install"], cwd=CDK_DIR)

    env = os.environ.copy()
    env["AWS_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_REGION"] = region
    env["CDK_DEFAULT_ACCOUNT"] = account_id

    cmd = [
        "npx",
        "cdk",
        "destroy",
        "-c", f"model={hub_id}",
        "-c", f"model_revision={revision}",
        "-c", f"instance_type={instance_type}",
        "-c", f"instance_count={instance_count}",
        "--force",
        "--progress",
        "events",
    ]
    run(cmd, cwd=CDK_DIR, env=env)


def clear_deployed_fields(config: dict[str, Any]) -> dict[str, Any]:
    sm = config.get("sagemaker")
    if isinstance(sm, dict):
        for key in (
            "endpoint_arn",
            "endpoint_config_name",
            "endpoint_status",
            "invoke_url",
            "image_uri",
            "model_data_s3_uri",
            "stack_arn",
            "deployed_at",
            "device",
        ):
            sm.pop(key, None)
        sm["endpoint_status"] = "Deleted"
    config.pop("smoke_test", None)
    return config


def confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="strands-decider SageMaker endpoint 삭제")
    parser.add_argument("--region", default=None, help="AWS 리전")
    parser.add_argument("--model", default=None, help="HF Hub id")
    parser.add_argument("--instance-type", default=None, help="배포 때 사용한 인스턴스 타입")
    parser.add_argument("-y", "--yes", action="store_true", help="확인 없이 삭제")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(SCRIPT_DIR)

    log("=" * 60)
    log("strands-decider-kr — SageMaker Endpoint 삭제")
    log("=" * 60)

    try:
        config = load_config()
        region, hub_id, instance_type, instance_count, revision = resolve_deploy_params(
            config, args
        )
        stack_name = endpoint_name_for(hub_id)

        sts = boto3.client("sts", region_name=region)
        account_id = sts.get_caller_identity()["Account"]

        log(f"region      : {region}")
        log(f"account_id  : {account_id}")
        log(f"model       : {hub_id}")
        log(f"instance    : {instance_type} x{instance_count}")
        log(f"stack       : {stack_name}")

        if not stack_exists(region, stack_name):
            log(f"스택이 없거나 이미 삭제됨: {stack_name}")
            config = clear_deployed_fields(config)
            # keep identifying fields for reinstall
            sm = config.setdefault("sagemaker", {})
            sm["stack_name"] = stack_name
            sm["endpoint_name"] = stack_name
            sm["instance_type"] = instance_type
            sm["instance_count"] = instance_count
            sm["cdk_context"] = {
                "model": hub_id,
                "model_revision": revision,
                "instance_type": instance_type,
                "instance_count": instance_count,
            }
            config["region"] = region
            config["account_id"] = account_id
            config.setdefault("model", {})["hub_id"] = hub_id
            save_config(config)
            log("config.json 정리 완료")
            return 0

        if not confirm(
            f"스택 '{stack_name}' 과 SageMaker endpoint를 삭제할까요?",
            assume_yes=args.yes,
        ):
            log("취소됨")
            return 1

        destroy_stack(
            region=region,
            account_id=account_id,
            hub_id=hub_id,
            revision=revision,
            instance_type=instance_type,
            instance_count=instance_count,
        )

        config = load_config()
        config["region"] = region
        config["account_id"] = account_id
        config.setdefault("model", {})["hub_id"] = hub_id
        config.setdefault("model", {})["revision"] = revision
        config.setdefault("model", {})["served_name"] = hub_id.split("/")[-1]
        sm = config.setdefault("sagemaker", {})
        sm["stack_name"] = stack_name
        sm["endpoint_name"] = stack_name
        sm["instance_type"] = instance_type
        sm["instance_count"] = instance_count
        sm["deploy_source"] = "deploy/sagemaker"
        sm["cdk_context"] = {
            "model": hub_id,
            "model_revision": revision,
            "instance_type": instance_type,
            "instance_count": instance_count,
        }
        config = clear_deployed_fields(config)
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
    log("삭제 완료")
    log("=" * 60)
    log(f"stack   : {stack_name}")
    log(f"config  : {CONFIG_PATH}")
    log("재설치   : python installer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
