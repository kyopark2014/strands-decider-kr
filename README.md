# Strands Decider

AWS SageMaker에 **Strands Decider E4B Gemma 4 v2** 를 배포하고, 한국어 테스트 케이스로 호출하는 프로젝트입니다.

Upstream: [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) (Apache 2.0)  
v2 발표: [Introducing Strands Decider v2](https://strandsagents.com/blog/introducing-strands-decider-v2/)  
배경 사례: [Amazon saves millions by replacing Sonnet with Strands Decider](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/)

## 목차

- [한 줄 요약](#한-줄-요약)
- [시스템 구성도](#시스템-구성도)
  - [전체 아키텍처](#전체-아키텍처)
  - [요청 흐름](#요청-흐름)
  - [설치 시 CDK가 만드는 것](#설치-시-cdk가-만드는-것)
- [디렉터리 구조](#디렉터리-구조)
- [사전 준비](#사전-준비)
- [설치 (installer.py)](#설치-installerpy)
  - [기본값](#기본값)
  - [실행](#실행)
  - [설치 단계](#설치-단계)
  - [설치 후 확인](#설치-후-확인)
- [실행 (run_strands_devider.py)](#실행-run_strands_deviderpy)
- [테스트 케이스 (testcases.json)](#테스트-케이스-testcasesjson)
- [실행 결과](#실행-결과)
  - [유형 설명](#유형-설명)
  - [케이스별 결과 요약](#케이스별-결과-요약)
  - [결과 분석](#결과-분석)
- [로컬 서버 테스트 (선택)](#로컬-서버-테스트-선택)
- [삭제 (uninstaller.py)](#삭제-uninstallerpy)
- [config.json](#configjson)
- [질문 유형 (API)](#질문-유형-api)
- [비용](#비용)
- [운영 팁](#운영-팁)
- [빠른 시작 체크리스트](#빠른-시작-체크리스트)
- [Reference](#reference)

## 한 줄 요약

| 항목 | 내용 |
| --- | --- |
| 모델 | `StrandsAgents/strands-decider-E4B-gemma4-v2-2611` |
| 서빙 이미지 | CDK `image=build` (PyPI `strands-decider==0.4.0`; DLC는 v2 미지원) |
| 배포 | SageMaker Real-time Endpoint (`deploy/sagemaker` CDK) |
| 기본 인스턴스 | `ml.g5.xlarge` (A10G 24GB, CUDA) |
| 리전 기본값 | `us-west-2` |
| 호출 | `boto3` `invoke_endpoint` ← `run_strands_devider.py` |
| 케이스 | `testcases.json` (한국어 noul / choice / score 10개) |

Strands Decider는 LLM처럼 텍스트를 생성하지 않고, **예/아니오·선택·점수**만 빠르게 판단하는 decision model(System One)입니다. 이 저장소는 그 모델을 AWS에 올리고 한국어 시나리오로 검증하는 운영 패키지입니다.

## 시스템 구성도

### 전체 아키텍처

```mermaid
flowchart LR
  subgraph Local["로컬 (strands-decider-kr)"]
    INST["installer.py"]
    RUN["run_strands_devider.py"]
    UN["uninstaller.py"]
    CFG["config.json"]
    TC["testcases.json"]
    CDK["deploy/sagemaker<br/>AWS CDK"]
  end

  subgraph AWS["AWS us-west-2"]
    S3["S3<br/>모델 + base weights"]
    CB["CodeBuild<br/>HF → S3 sync"]
    SM["SageMaker Endpoint<br/>ml.g5.xlarge"]
    DLC["ECR Image<br/>Dockerfile + strands-decider 0.4.0"]
    CW["CloudWatch Logs"]
  end

  HF["Hugging Face Hub<br/>E4B + Gemma 4 base"]

  INST --> CDK
  CDK --> CB
  CB --> HF
  CB --> S3
  CDK --> SM
  S3 --> SM
  DLC --> SM
  SM --> CW
  INST --> CFG
  RUN --> CFG
  RUN --> TC
  RUN -->|"InvokeEndpoint<br/>/invocations"| SM
  UN --> CDK
  UN --> CFG
```

### 요청 흐름

```mermaid
sequenceDiagram
  participant User as 사용자/스크립트
  participant Runner as run_strands_devider.py
  participant CFG as config.json
  participant TC as testcases.json
  participant SM as SageMaker Endpoint
  participant Model as strands-decider E4B

  User->>Runner: python run_strands_devider.py
  Runner->>CFG: region, endpoint_name 로드
  Runner->>TC: cases[] 로드
  loop 각 테스트 케이스
    Runner->>SM: invoke_endpoint(JSON, sort_keys=True)
    SM->>Model: POST /invocations
    Model-->>SM: answers + latency_ms
    SM-->>Runner: SystemOneResponse
    Runner-->>User: instructions/criteria/result 출력
  end
```

### 설치 시 CDK가 만드는 것

1. **S3 버킷** — Hugging Face에서 E4B 어댑터 + Gemma 4 base를 받아 저장 (CodeBuild)
2. **SageMaker Model** — DLC 이미지 + S3 `ModelDataSource` (`HF_HUB_OFFLINE=1`)
3. **EndpointConfig / Endpoint** — `ml.g5.xlarge` × 1, CUDA
4. **IAM Role** — 버킷 읽기, 이미지 pull, CloudWatch 로그
5. 배포 후 `installer.py`가 **endpoint ARN, Invoke URL, 스모크 결과**를 `config.json`에 기록

엔드포인트 API는 로컬 `POST /v1/systemone`과 동일한 본문이며, SageMaker 경로는 `POST /invocations` / `GET /ping`입니다.

## 디렉터리 구조

```text
strands-decider-kr/
├── README.md
├── installer.py              # SageMaker 배포 + config.json 생성/갱신
├── uninstaller.py            # CDK destroy + config 정리
├── run_strands_devider.py    # config + testcases → endpoint 호출 (단독 실행)
├── testcases.json            # 한국어 테스트 케이스 10개
├── test_systemone_examples.py# 로컬 serve URL용 러너 (testcases.json 사용)
├── config.json               # 배포 결과 (gitignore, installer가 생성)
└── deploy/sagemaker/         # 공식 strands-decider SageMaker CDK 앱
    ├── bin/
    ├── lib/
    ├── scripts/fetch-model.ts
    ├── cdk.json
    └── package.json
```

## 사전 준비

| 요구사항 | 설명 |
| --- | --- |
| AWS 자격 증명 | 계정에 SageMaker / CloudFormation / IAM / S3 / CodeBuild 권한 |
| Node.js 22+ | `npx cdk` 실행 (`deploy/sagemaker`) |
| Python 3.10+ | `boto3` 설치 |
| SageMaker 쿼터 | `ml.g5.xlarge for endpoint usage` ≥ 1 (Service Quotas) |

```bash
pip install boto3
# Node.js 22+ 필요
cd deploy/sagemaker && npm install && cd ../..
```

CDK bootstrap은 `installer.py`가 리전별로 없을 때 자동 실행합니다.

## 설치 (installer.py)

`deploy/sagemaker` CDK로 엔드포인트를 만들고, 결과를 `config.json`에 저장합니다.  
파일이 없어도 **기본값으로 새로 생성**합니다.

### 기본값

| 키 | 기본값 |
| --- | --- |
| `model.hub_id` | `StrandsAgents/strands-decider-E4B-gemma4-v2-2611` |
| `sagemaker.instance_type` | `ml.g5.xlarge` |
| `sagemaker.instance_count` | `1` |
| `region` | AWS 세션 리전, 없으면 `us-west-2` |

### 실행

```bash
python installer.py

# 옵션
python installer.py --region us-west-2
python installer.py --instance-type ml.g5.xlarge
python installer.py --model StrandsAgents/strands-decider-E4B-gemma4-v2-2611
python installer.py --skip-smoke
```

### 설치 단계

1. `config.json` 로드(없으면 생성) + CLI/기본값 반영 후 저장  
2. `deploy/sagemaker`에서 `npm install` (필요 시)  
3. `npx cdk bootstrap aws://<account>/<region>` (필요 시)  
4. `npx cdk deploy -c model=... -c instance_type=ml.g5.xlarge --require-approval never`  
5. CloudFormation Outputs / SageMaker Describe로 `config.json` 갱신  
6. 스모크 테스트 1건 (`noul` 긴급성) 후 `smoke_test` 필드 저장  

배포에는 모델 다운로드 + 엔드포인트 기동으로 **약 10~20분** 걸릴 수 있습니다.

### 설치 후 확인

```bash
aws sagemaker describe-endpoint \
  --endpoint-name strands-decider-e4b-gemma4-v2-2611 \
  --region us-west-2 \
  --query EndpointStatus
# InService
```

## 실행 (run_strands_devider.py)

`config.json` + `testcases.json`만으로 **단독 실행**합니다. (`test_systemone_examples.py` import 없음)

```bash
python run_strands_devider.py
python run_strands_devider.py --list
python run_strands_devider.py --only 1,4,8
python run_strands_devider.py --config ./config.json --cases ./testcases.json
```

동작 요약:

1. `config.json`에서 `region`, `sagemaker.endpoint_name`, `notes.sort_json_keys` 읽기  
2. `testcases.json`의 `cases[]` 로드  
3. 각 케이스의 `request`를 `sagemaker-runtime.invoke_endpoint`로 전송  
4. `instructions` / `criteria` / `result` / 클라이언트·서버 지연시간 출력  

JSON은 기본적으로 **`sort_keys=True`** 로 직렬화합니다. (키 순서만 달라도 확률이 흔들릴 수 있음 — Amazon 사례 블로그)

출력 예:

```text
endpoint=strands-decider-e4b-gemma4-v2-2611  region=us-west-2  instance=ml.g5.xlarge
---- is_urgent (noul) ----
instructions: 이 메시지에 긴급함이 느껴지나요?
result: noul=0.9560  → 예(true) 쪽
실행시간: 646.91 ms (클라이언트) / 105.84 ms (서버)
```

## 테스트 케이스 (testcases.json)

한국어 시나리오 **10개**. 형식:

```json
{
  "version": 1,
  "description": "...",
  "cases": [
    {
      "name": "01_noul_urgency",
      "description": "...",
      "request": {
        "state": "...",
        "questions": {
          "is_urgent": {
            "type": "noul",
            "instructions": "이 메시지에 긴급함이 느껴지나요?"
          }
        }
      }
    }
  ]
}
```

| # | 이름 | 유형 | 내용 |
| --- | --- | --- | --- |
| 01 | noul_urgency | noul | 정산 실패 메시지의 긴급성 |
| 02 | choice_team_routing | choice | 결제 / 영업 / 리테일 라우팅 |
| 03 | score_frustration | score | 좌절도 (차분함→절망적임) |
| 04 | multi_question | noul+choice+score | 한 state에 질문 3개 |
| 05 | language_detect | choice | 짧은 문구 언어 분류 |
| 06 | factual_check | noul | strawberry r 개수 |
| 07 | sentiment | score | 문서 긍정성 |
| 08 | tool_gate | choice | 에이전트 다음 도구 선택 |
| 09 | needs_human | noul | 사람 에스컬레이션 여부 |
| 10 | answer_quality | score+noul | 답변 품질·실행 가능성 |

## 실행 결과

아래는 SageMaker Endpoint (`strands-decider-E4B-gemma4-v2-2611`, `ml.g5.xlarge`, 커스텀 이미지 `strands-decider==0.4.0`)에 대해  
`python run_strands_devider.py` 로 10개 케이스를 돌린 **v2 실측** 결과입니다.

### 유형 설명

| 유형 | 의미 | 이 프로젝트에서의 역할 |
| --- | --- | --- |
| **noul** | 예/아니오 확률 `P(true)` (0~1) | 긴급성, 사실 확인, 사람 에스컬레이션, 실행 가능성 |
| **choice** | N개 옵션 중 1개 + 확률·confidence | 팀 라우팅, 언어 감지, 툴 게이트 |
| **score** | 순서 척도의 기대값(연속값) | 좌절도, 긍정성, 답변 품질 |
| **multi** | 한 `state`에 noul+choice+score를 묶음 | 티켓 한 건을 한 요청으로 triage |
| **score+noul** | 품질 점수와 yes/no를 함께 | 에이전트 답변 평가 |

### 케이스별 결과 요약

| # | 유형 | 시나리오 | 결과 | 서버 latency |
| --- | --- | --- | --- | --- |
| 01 | noul — 긴급성 | 정산 3일 실패 | **noul=0.9965** → 예 | 73.7 ms |
| 02 | choice — 팀 라우팅 | 동일 티켓 | **결제** (1.000, conf 1.000) | 73.8 ms |
| 03 | score — 좌절도 | 동일 티켓 | **score≈1.01 ≈ 좌절함** (conf 0.852) | 77.4 ms |
| 04 | multi — noul+choice+score | 한 요청에 3질문 | 긴급 0.997 / **결제** / 좌절≈1.01 | 136.6 ms |
| 05 | choice — 언어 감지 | `sihamba ngokushesha` | **줄루어** (0.978, conf 0.966) | 73.8 ms |
| 06 | noul — strawberry r 개수 | `strawberry`에 r이 3개? | **noul=0.558** → 예 쪽 | 73.7 ms |
| 07 | score — 긍정성 | 극찬 리뷰 | **score≈3.999 ≈ 매우 긍정적** (conf 0.982) | 72.9 ms |
| 08 | choice — 툴 게이트 | 구독 해지·환불 | **환불생성** (0.890, conf 0.854) | 74.7 ms |
| 09 | noul — 사람 에스컬레이션 | 해킹·법적 위협 | **noul=0.9978** → 예 | 75.3 ms |
| 10 | score+noul — 답변 품질 | 비밀번호 재설정 Q&A | score≈2.36 ≈ 수용 가능 / **actionable 0.964** | 137.7 ms |

### 결과 분석

**v1 → v2에서 눈에 띄는 변화**

- **01 긴급성**: 0.956 → **0.997** (더 확신).
- **06 strawberry**: v1은 **0.457(아니오 쪽, 오답)** 이었는데 v2는 **0.558(예 쪽, 정답 방향)**. 여전히 애매하니 문자 세기에는 폴백을 유지하세요.
- **07 긍정성**: ≈3.61 → **≈3.999** (거의 만점, conf 0.98).
- **서버 latency**: 단건 ~103 ms → **~73–77 ms**.

**잘 맞는 케이스**

- **01 / 02 / 03 / 09**: 티켓 triage·에스컬레이션에 바로 쓰기 좋습니다.
- **04 multi**: 3질문 ~137 ms — state 1회 인코딩 이점이 유지됩니다.
- **05 / 08**: 언어·툴 게이트도 기대 방향.

**해석에 주의할 케이스**

- **06**: 정답 방향이지만 confidence가 낮습니다. 규칙/LLM 폴백 권장.
- **10**: score 질량이 `명확하고 완전함`(0.535)과 `수용 가능함`(0.302)에 갈라져 conf 0.48. **is_actionable**과 함께 보세요.

**지연시간**

| 지표 | 값 |
| --- | --- |
| 총 (클라이언트) | 약 **2.7 s** / 10건 |
| 평균 (클라이언트) | **~272 ms** |
| 서버 단건 | 대개 **~73–77 ms** |
| 서버 multi(3질문) | **~137 ms** |
| 첫 요청 | 클라이언트 ~576 ms (웜업/연결 포함 가능) |

**실무 권장**

1. 라우팅·툴 게이트·에스컬레이션은 confidence ≥ 0.9일 때 자동 적용.  
2. 문자 세기·열린 추론은 Decider만으로 결정하지 말고 폴백.  
3. 티켓 triage는 **04 multi**처럼 한 요청에 묶어 RTT를 절약.  
4. JSON은 `sort_keys=True`로 보내 회귀 비교를 안정화.  
5. **배포 이미지**: 공식 DLC는 아직 v1 config만 로드하므로, 이 프로젝트는 `-c image=build`(PyPI `strands-decider==0.4.0`)를 기본으로 씁니다.

## 로컬 서버 테스트 (선택)

로컬에 모델을 띄운 뒤 HTTP로 같은 `testcases.json`을 돌릴 때:

```bash
# 별도 터미널
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v2-2611 --port 8000

python test_systemone_examples.py --base-url http://localhost:8000
python test_systemone_examples.py --cases ./testcases.json --only 1,2 --list
```

## 삭제 (uninstaller.py)

CDK 스택과 SageMaker 엔드포인트를 제거하고, `config.json`의 배포 결과 필드를 정리합니다.  
재설치에 필요한 `model` / `instance_type` / `cdk_context`는 남깁니다.

```bash
python uninstaller.py          # 확인 프롬프트
python uninstaller.py --yes    # 확인 없이 삭제

# 배포 때와 동일한 모델/인스턴스를 명시할 수도 있음
python uninstaller.py \
  --model StrandsAgents/strands-decider-E4B-gemma4-v2-2611 \
  --instance-type ml.g5.xlarge \
  --yes
```

삭제 후:

- CloudFormation 스택 `strands-decider-e4b-gemma4-v2-2611` 제거  
- Endpoint / EndpointConfig / Model / S3 모델 버킷 등 스택 리소스 삭제  
- `config.json`의 `endpoint_arn`, `invoke_url`, `smoke_test` 등 제거, `endpoint_status=Deleted`  

> 엔드포인트는 **요청이 없어도 인스턴스 시간당 과금**됩니다. 실험이 끝나면 `uninstaller.py`로 지우세요.

## config.json

`installer.py`가 생성·갱신합니다. **gitignore 대상**이라 커밋되지 않습니다.

주요 필드:

```json
{
  "project": "strands-decider-kr",
  "account_id": "262976740991",
  "region": "us-west-2",
  "model": {
    "hub_id": "StrandsAgents/strands-decider-E4B-gemma4-v2-2611",
    "revision": "main",
    "served_name": "strands-decider-E4B-gemma4-v2-2611"
  },
  "sagemaker": {
    "stack_name": "strands-decider-e4b-gemma4-v2-2611",
    "endpoint_name": "strands-decider-e4b-gemma4-v2-2611",
    "endpoint_arn": "arn:aws:sagemaker:us-west-2:...:endpoint/...",
    "instance_type": "ml.g5.xlarge",
    "instance_count": 1,
    "device": "cuda",
    "invoke_url": "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/.../invocations",
    "image_uri": ".../strands-decider:serve-sagemaker-cuda-v1",
    "model_data_s3_uri": "s3://.../main/",
    "cdk_context": { "model": "...", "instance_type": "ml.g5.xlarge" }
  },
  "notes": {
    "invoke_permission": "sagemaker:InvokeEndpoint",
    "sort_json_keys": true
  }
}
```

호출 IAM: `sagemaker:InvokeEndpoint` on the endpoint ARN.

## 질문 유형 (API)

요청 본문 (`ContentType: application/json`):

| type | 의미 | 응답 |
| --- | --- | --- |
| `noul` | 예/아니오 확률 | `noul` = P(true) ∈ [0,1] |
| `choice` | N개 중 하나 | `choice`, `probabilities`, `confidence` |
| `score` | 순서 척도 기대값 | `score`, `legend`, `probabilities`, `confidence` |

```python
import json, boto3

runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")
body = {
    "state": "도와주세요! 정산금 지급이 3일째 계속 실패하고 있어요!",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "이 메시지에 긴급함이 느껴지나요?",
        }
    },
}
resp = runtime.invoke_endpoint(
    EndpointName="strands-decider-e4b-gemma4-v2-2611",
    ContentType="application/json",
    Body=json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8"),
)
print(json.loads(resp["Body"].read()))
```

## 비용

현재 배포 기준(온디맨드 공시가):

| 항목 | 값 |
| --- | --- |
| 리전 | `us-west-2` |
| 인스턴스 | `ml.g5.xlarge` × **1** (Real-time Hosting) |
| 단가 | **$1.408 / instance-hour** |
| 과금 방식 | Endpoint가 `InService`인 **매 시간** (요청 0건이어도 동일) |

단가는 AWS Price List API (`AmazonSageMaker`, `ml.g5.xlarge-Hosting`, US West Oregon) 조회값이며, us-east-1과 동일합니다.

### 월 비용 예측 (compute)

| 시나리오 | 계산 | 예상 비용 |
| --- | --- | --- |
| **24/7 상시 가동 (권장 월 추정)** | `$1.408 × 24 × 30` | **약 $1,014 / 월** |
| 평균 월 730시간 | `$1.408 × 730` | **약 $1,028 / 월** |
| 업무시간만 8h×22일 | `$1.408 × 176` | 약 **$248 / 월** (수동 on/off 또는 스케줄 삭제 가정) |
| 실험 1주일 24/7 | `$1.408 × 24 × 7` | 약 **$237 / 주** |
| 하루만 켜 둠 | `$1.408 × 24` | 약 **$33.8 / 일** |

> Real-time Endpoint는 **scale-to-zero가 기본이 아닙니다.** `installer.py`로 올린 뒤 방치하면 위 **~$1,014–1,028/월**이 그대로 청구됩니다. 실험이 끝나면 `python uninstaller.py --yes`로 삭제하세요.

### 부가 비용 (상대적으로 작음)

| 항목 | 예상 |
| --- | --- |
| S3 모델 artifact (E4B + base, 대략 수십 GB) | Standard 기준 대략 **수 달러/월** 이하 |
| CloudWatch Logs | 테스트·짧은 보관이면 **$1 미만** 수준 |
| 데이터 전송 (같은 리전 Invoke) | 보통 무시 가능 |
| CodeBuild (배포 1회) | 모델 sync 시 일회성, 월 고정비 아님 |

**실무적으로 월 예산의 거의 전부 SageMaker Hosting 인스턴스 시간**입니다.

### 처리량 관점 (참고)

Amazon 사례·실측 기준 서버 단건 ~100–640 ms대입니다. 대략 **요청당 150 ms**, concurrency 1이면 이론상 인스턴스당 시간당 ~24,000건 수준입니다.

| 가정 | 인스턴스 시간당 비용 | 요청당 단가 (대략) |
| --- | --- | --- |
| 풀가동 ~20,000 req/h | $1.408 | **~$0.00007 / 요청** |
| 여유 두고 ~5,600 req/h (블로그) | $1.408 | **~$0.00025 / 요청** |
| 거의 idle (시간당 10건) | $1.408 | **~$0.14 / 요청** |

트래픽이 낮을수록 요청당 단가가 급등하므로, **상시 API가 아니면 끄고 쓰는 것**이 비용 효율의 핵심입니다.

### 절감 팁

1. 사용 후 즉시 `uninstaller.py`로 스택 삭제  
2. 필요 시에만 `installer.py`로 다시 배포 (콜드 기동 수~십수 분)  
3. 장기 상시라면 Savings Plans / 더 저렴한 `ml.g6.xlarge`(L4) 검토 — 다만 본 프로젝트 기본·검증은 `ml.g5.xlarge`  
4. 오토스케일 min=0 구성은 Real-time에서 제약이 크므로, 이 스택에서는 **삭제 = 과금 중단**으로 보는 것이 단순합니다

## 운영 팁

- **동시성**: DLC 기본 `DECIDER_MAX_CONCURRENCY=1` 유지 권장 (버스트 시 OOM 방지).  
- **JSON 키 순서**: 비교·회귀 테스트 시 `sort_keys=True`.  
- **용량 부족**: CDK context `fallback_instance_types` (예: `ml.g6e.xlarge,ml.g6.xlarge`) — 필요 시 `installer`/`cdk_context`에 추가.  
- **비용**: 자세한 월 추정은 [비용](#비용). 유휴 엔드포인트가 가장 큰 낭비이므로 미사용 시 삭제.  
- **로그**:

```bash
aws logs tail /aws/sagemaker/Endpoints/strands-decider-e4b-gemma4-v2-2611 \
  --region us-west-2 --since 1h --follow
```


## 빠른 시작 체크리스트

```bash
# 1) 배포
python installer.py

# 2) 케이스 실행
python run_strands_devider.py --list
python run_strands_devider.py

# 3) 실험 종료 후 삭제 (과금 중단)
python uninstaller.py --yes
```

## Reference

- Upstream: https://github.com/strands-labs/strands-decider  
- SageMaker 배포 가이드: https://github.com/strands-labs/strands-decider/tree/main/deploy/sagemaker  
- DLC: https://aws.github.io/deep-learning-containers/strands-decider/  
- 모델 (E4B v2): https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v2-2611  
- Decider v2 블로그: https://strandsagents.com/blog/introducing-strands-decider-v2/  
- Amazon 사례 블로그: https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/  
- Decision Model 8종 성능 비교: Jev, Kev, Liquid d1 중 무엇을 선택해야 할까?: https://www.arizekorea.com/insights/post/decision-model-benchmark-jev-kev-liquid-d1-2LXJin4juMAAhxr
- strands-labs / strands-decider: https://github.com/strands-labs/strands-decider
- guardrail-bench — 한국어 금융권 컨텐츠 필터링 가드 벤치마크: https://github.com/hsr87/korean-guardrail-bench
- Amazon saves millions by replacing Sonnet 4.6 with Strands Decider: https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/
