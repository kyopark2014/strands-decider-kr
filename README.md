# Strands Decider — 에이전트용 소형 결정 모델 (System One)

> 조사일: 2026-11-11 (업데이트: 2026-11-12, Amazon 사례 + E4B SageMaker/EC2 배포 검토) · 출처: [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) (Apache 2.0)  
> 관련 노트: [[Jev — TypeSafe System One Model]]

## 목차

- [한 줄 요약](#한-줄-요약)
- [LLM·기존 분류기와의 비교](#llm기존-분류기와의-비교)
- [질문 유형 3가지](#질문-유형-3가지)
- [사용 방법 (CLI · 서버)](#사용-방법-cli--서버)
- [아키텍처](#아키텍처)
- [모델 라인업](#모델-라인업)
- [성능](#성능)
- [활용 사례](#활용-사례)
- [Strands Agents 연동 예시](#strands-agents-연동-예시)
- [배포 옵션](#배포-옵션)
  - [E4B AWS 배포 검토 (SageMaker vs EC2)](#e4b-aws-배포-검토-sagemaker-vs-ec2)
  - [SageMaker Real-time Endpoint (권장)](#sagemaker-real-time-endpoint-권장)
  - [EC2 직접 배포](#ec2-직접-배포)
  - [비교와 선택 기준](#비교와-선택-기준)
- [커스터마이징·학습·연구 기록](#커스터마이징학습연구-기록)
- [커뮤니티 사례](#커뮤니티-사례)
- [Amazon 사례 — Sonnet 4.6을 Decider로 교체 (98% 절감)](#amazon-사례--sonnet-46을-decider로-교체-98-절감)
- [직접 확인하지 못한 부분](#직접-확인하지-못한-부분)
- [참고 링크](#참고-링크)

## 한 줄 요약

Strands Decider는 Strands Agents 팀이 공개한 오픈소스 **결정 모델(decision model, "System One" 모델)** 입니다. LLM처럼 텍스트를 생성하지 않고 **선택지 중 하나를 고르거나 척도로 점수를 매기는 일**만 하며, 모든 답에 **보정된 신뢰도(calibrated confidence)** 를 함께 돌려줍니다. 에이전트 워크플로 안의 반복적인 판단을 LLM보다 빠르고 저렴하게 처리하는 것이 목적입니다.

- 저장소: `strands-labs/strands-decider` (2026-09-29 공개, Star 약 1,100개)
- 모델: Hugging Face `StrandsAgents` 조직이 원 저장소이며, v1 모델 5종은 `amazon` 조직에도 같은 이름으로 올라가 있습니다.
- 설치: `pip install strands-decider`

## LLM·기존 분류기와의 비교


| 구분    | LLM     | Strands Decider       | 기존 분류 모델 |
| ----- | ------- | --------------------- | -------- |
| 출력    | 자유 텍스트  | 선택지 선택 / 점수           | 고정 레이블   |
| 속도    | 느림      | 빠름 (질문당 약 100\~250ms) | 매우 빠름    |
| 별도 학습 | 불필요     | 불필요 (요청 시 선택지 정의)     | 작업마다 필요  |
| 신뢰도   | 제공하지 않음 | 보정된 신뢰도 제공            | 모델마다 다름  |


학습 때 보지 못한 짧은 분류 작업에서 **신뢰도 0.9 이상인 답은 약 95%가 정답**이었습니다. 따라서 "0.9 미만이면 사람이 확인하거나 상위 LLM으로 넘기는" 방식이 권장됩니다.

## 질문 유형 3가지


| 유형       | 용도              | 예시 (결과)                                             |
| -------- | --------------- | --------------------------------------------------- |
| `noul`   | 예/아니오 확률 (0\~1) | "긴급한 내용인가?" → 0.858                                 |
| `choice` | N개 중 하나 선택      | "담당 팀은?" billing/sales/retail → billing (신뢰도 0.659) |
| `score`  | 순서가 있는 척도       | "얼마나 화가 났나?" calm/frustrated/depressed → 1.03       |


여러 질문을 한 번에 보낼 수 있습니다. 상태 텍스트는 **한 번만 읽고**, 질문마다 해당 질문의 토큰만 추가로 처리하므로 같은 텍스트에 질문을 여러 개 붙여도 비용이 거의 늘지 않습니다.

## 사용 방법 (CLI · 서버)

```bash
pip install strands-decider

strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000

curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}'
# → {"answers":{"is_urgent":{"type":"noul","noul":0.8581}},"usage":{"input_tokens":86,"output_tokens":1},"latency_ms":489.01}
```

- 디바이스: `--device cuda | mps | cpu | mlx`. Apple Silicon에서는 MLX가 MPS보다 1.4\~1.6배 빠릅니다.
- 이미지: Qwen3.5 모델은 `--vision` 옵션과 `pip install "strands-decider[vision]"`(transformers 5.18 이상)이 있으면 base64 이미지를 상태로 받습니다. 예: "서명란이 채워졌나?"

## 아키텍처

- 사전학습된 디코더 LLM 본체(**Qwen3.5-2B-Base**)에서 **언어 모델 헤드를 제거**합니다. 그래서 텍스트를 생성할 수 없습니다.
- 그 자리에 약 100만 파라미터의 **pointer head**를 붙입니다. 이 헤드는 `<answer>` 위치의 hidden state와 각 선택지 마지막 토큰의 hidden state를 비교해 점수를 매깁니다.
- **forward pass 한 번**으로 끝나며 디코딩 루프가 없습니다. 본체는 rank-16 LoRA로 조정하고, 헤드는 fp32로 실행합니다.
- 헤드에 선택지별 파라미터가 없습니다. 그래서 선택지 개수에 제한이 없고, "첫 번째 선택지가 보통 정답" 같은 위치 편향을 학습할 수 없습니다.
- 세 질문 유형(`noul`/`choice`/`score`)은 모두 같은 masked softmax에서 나오며, 결과를 읽는 방식만 다릅니다.
- 이전 1세대는 slot head 방식이었고 성능이 크게 떨어졌습니다. 현재 구조는 2세대입니다.
- v1 기준 모델은 246,678행의 균형 데이터(코드 작업, 정책 적용 규칙, 공개 데이터셋 추가)로 학습했습니다. teacher는 gemma-4-31B-it이며, 학습 3회분을 평균(soup)한 뒤 보정했습니다.

## 모델 라인업

이름 규칙: `strands-decider-<크기>-<기반모델>-v<N>-<YYMM>`


| 모델                         | JevBench (231문항) | Brier 점수 | GPU 메모리 (bf16) | 지연 (중앙값)                 |
| -------------------------- | ---------------- | -------- | -------------- | ------------------------ |
| 2B-qwen3.5-v1-2610 (기준 모델) | 180              | 0.280    | –              | RTX 3090 115ms (v19 측정값) |
| E2B-gemma4-v1-2610         | 180              | 0.313    | 약 9GB          | H100 102ms               |
| E4B-gemma4-v1-2610         | 187              | 0.243    | 약 15GB         | H100 139ms               |
| 12B-gemma4-v1-2610         | 200              | 0.173    | 약 23GB         | H100 136ms               |
| 26B-A4B-gemma4-v1-2610     | 207              | 0.162    | 약 51GB         | H100 245ms               |


- 이전 모델인 `strands-decider-2B-hobson-v19`와 `-v21`도 계속 공개되어 있습니다.
- Gemma 4 모델은 아직 PyPI에 없어서 GitHub `main`에서 설치해야 하며, 텍스트만 지원합니다.
- 크기 선택 가이드: 정확도는 12B와 26B-A4B가 가장 높고, 보정은 26B-A4B가 더 좋습니다(메모리·지연은 12B의 약 2배). GPU를 다른 작업과 함께 쓴다면 E4B, 메모리가 가장 부족하거나 CPU에서 돌린다면 E2B가 적합합니다.

## 성능


| 항목                            | v1                    | v21                   | v19                   |
| ----------------------------- | --------------------- | --------------------- | --------------------- |
| JevBench 정확도                  | 0.779 (180/231)       | 0.762 (176/231)       | 0.723 (167/231)       |
| Brier 점수 / ECE                | 0.280 / 0.072         | 0.323 / 0.064         | 0.342 / 0.052         |
| 난이도별 정확도 (easy/standard/hard) | 1.000 / 0.931 / 0.586 | 1.000 / 0.931 / 0.550 | 1.000 / 0.875 / 0.505 |


- v1은 코드 작업 1,996건(학습에 쓰지 않은 데이터)에서 0.836을 기록했습니다.
- 지연(v19): RTX 3090 중앙값 115ms / p95 299ms, M3 Pro 웜 상태 중앙값 153ms.
- **주의**: 231문항은 적은 수라서, 단일 실행끼리 약 10문항 미만의 차이는 의미 있는 차이로 보기 어렵다고 README가 명시합니다.

## 활용 사례

- **모델 라우팅**: 작업에 맞는 LLM을 선택합니다.
- **도구 선택**: 에이전트가 다음에 호출할 도구를 결정합니다.
- **인자 검증**: 도구를 실행하기 전에 인자가 타당한지 확인합니다.
- **트리아지**: 들어온 요청을 담당 팀이나 큐로 분배합니다.
- **가드레일**: grounding 확인, 안전성·정책 분류를 수행합니다.
- **평가(Eval)**: 모델 출력의 품질과 답변의 적절성을 저렴하게 점수화합니다.
- **하이브리드 에이전트**: 어려운 판단은 LLM, 반복적인 판단은 Decider에 맡겨 비용과 지연을 줄입니다.

## Strands Agents 연동 예시

`examples/strands/tool_call_intervention.py`에 예제가 있습니다.

- Strands의 `InterventionHandler`에 `before_tool_call` 을 구현하고 에이전트에 전달합니다.
- 시나리오: 사용자가 "날씨 어때?"라고만 물었는데, 에이전트가 도시를 추측해 `get_weather`를 호출하려고 합니다.
- 도구가 실행되기 전에 Decider가 두 가지 예/아니오 질문에 답합니다.
  1. 인자 값이 사용자가 실제로 제공한 사실에 근거하는가?
  2. 사용자에게 확인하기 전에 도구를 호출하는 것은 너무 이른가?
- 결과에 따라 `Proceed`(실행) 또는 `Guide`(피드백과 함께 모델에 차례를 돌려줌)를 반환합니다. 이 경우 에이전트는 "어느 도시를 말씀하시나요?"라고 되묻습니다.
- 장점: 로컬 forward pass로 약 200ms 안에 판단하고, 대화가 외부로 나가지 않습니다. 판정이 **숫자**로 나오므로 코드의 `if` 문으로 분기할 수 있고, 에이전트가 말로 우길 여지가 없습니다.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
pip install -e . strands-agents
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
python examples/strands/tool_call_intervention.py   # Bedrock 자격증명 필요
```

## 배포 옵션


| 방식                | 내용                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| 로컬                | Linux CUDA, Mac(MPS/MLX), CPU에서 `strands-decider serve`를 실행합니다.                                        |
| Amazon SageMaker  | 공식 CDK(`deploy/sagemaker`) + DLC로 Real-time Endpoint. E4B 프로덕션의 기본 경로입니다.                              |
| EC2 직접            | GPU EC2에서 `strands-decider serve` 또는 동일 DLC를 직접 실행. Spot·커스텀 파이프라인용.                                   |
| AWS Lambda        | arm64 CPU에서 llama.cpp로 양자화한 **2B** 모델을 IAM 인증 Function URL 뒤에 둡니다. 서버리스지만 응답에 수 초가 걸립니다. E4B에는 부적합합니다. |
| AgentCore Runtime | 공식 가이드는 없지만 커뮤니티 사례가 있습니다(아래 참고).                                                                      |


### E4B AWS 배포 검토 (SageMaker vs EC2)

> 검토일: 2026-11-12 · 근거: [Amazon 사례 블로그](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/), [deploy/sagemaker/README.md](https://github.com/strands-labs/strands-decider/blob/main/deploy/sagemaker/README.md), [strands-decider DLC](https://aws.github.io/deep-learning-containers/strands-decider/)

Amazon 내부 티켓 라벨링에서 **E4B(Gemma 4)** 가 2B보다 명확히 좋아 채택되었고, 운영은 **SageMaker** `ml.g5.xlarge` 위에서 돌았습니다. 고객도 같은 경로를 그대로 쓸 수 있습니다.


| 항목             | 값                                                                   |
| -------------- | ------------------------------------------------------------------- |
| 모델 Hub id      | `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`                  |
| GPU 가중치 (bf16) | 약 **15 GB** (블로그 실측 메모리 \~**19 GiB**)                               |
| 권장 인스턴스        | 24 GB GPU — `ml.g6.xlarge`(L4, CDK 기본) 또는 `ml.g5.xlarge`(A10G, 블로그) |
| 블로그 지연         | 요청당 약 **640 ms** (인스턴스 1대 이론 \~5,600건/시간)                           |
| CPU 대안         | fp32 약 30 GB → `ml.m5.4xlarge` 등 (느림)                               |


### SageMaker Real-time Endpoint (권장)

공식 스택이 S3에 모델+base를 넣고, GPU 패밀리에서는 게시된 DLC를 쓰며, 엔드포인트는 인터넷 없이(`HF_HUB_OFFLINE=1`) 기동합니다. 요청 본문은 `POST /v1/systemone`과 동일하고, SageMaker 경로는 `POST /invocations` / `GET /ping`입니다.

```bash
git clone https://github.com/strands-labs/strands-decider
cd strands-decider/deploy/sagemaker
npm install
npx cdk bootstrap aws://<account>/<region>

npx cdk deploy \
  -c model=StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  -c instance_type=ml.g6.xlarge
```

용량 부족 시 fallback:

```bash
npx cdk deploy \
  -c model=StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  -c instance_type=ml.g6.xlarge \
  -c fallback_instance_types=ml.g6e.xlarge,ml.g5.xlarge
```

호출 예:

```python
import json, boto3

runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="strands-decider-e4b-gemma4-v1-2610",
    ContentType="application/json",
    Body=json.dumps(
        {
            "state": "도와주세요! 정산금 지급이 3일째 계속 실패하고 있어요!",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "이 메시지에 긴급함이 느껴지나요?",
                }
            },
        },
        sort_keys=True,  # 키 순서가 확률을 흔들 수 있음 (블로그: 최대 ~0.19)
    ),
)
print(json.loads(response["Body"].read())["answers"])
```

운영 포인트:

- `DECIDER_MAX_CONCURRENCY=1`**(기본)** 유지. 질문 배치는 요청 내부에서 이미 되고, 동시 요청을 풀면 E4B에서 OOM·RPS 악화가 관측됨.
- IAM `sagemaker:InvokeEndpoint`로만 접근. 컨테이너 자체 인증 없음.
- 블로그 패턴: SQS → Lambda → Endpoint + AppConfig 3-way(`BEDROCK` / `DUAL` / `DECIDER`) + CloudWatch.
- DLC 이미지: `public.ecr.aws/deep-learning-containers/strands-decider:serve-sagemaker-cuda-v1` (주간 보안 패치).
- 제거: `npx cdk destroy -c model=StrandsAgents/strands-decider-E4B-gemma4-v1-2610` (유휴 시간도 과금).

장점: IAM·VPC 격리·오토스케일·오프라인 가중치·CDK 재현.  
단점: 유휴 인스턴스 과금, 배포/기동 시간, 컨테이너 커스터마이즈는 제한적(`-c image=build`로 Dockerfile 빌드 가능).

### EC2 직접 배포

공식 EC2 CDK는 없고, 같은 `serve` API 또는 SageMaker DLC를 GPU EC2에 올리는 DIY입니다. 인스턴스: `g6.xlarge` / `g5.xlarge`(24 GB VRAM).

```bash
# Gemma 4는 PyPI 패키지보다 main 권장
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"

huggingface-cli download StrandsAgents/strands-decider-E4B-gemma4-v1-2610

strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --host 0.0.0.0 --port 8000 --device cuda --max-concurrency 1
```

DLC를 EC2에서 재사용 (가중치는 호스트에 미리 두고 마운트, 오프라인):

```bash
docker run --gpus all -p 127.0.0.1:8080:8080 \
  -v /opt/models/e4b:/opt/ml/model:ro \
  -e HF_HUB_OFFLINE=1 \
  -e DECIDER_MAX_CONCURRENCY=1 \
  public.ecr.aws/deep-learning-containers/strands-decider:serve-sagemaker-cuda-v1
```

앞에 둘 것: ALB/타겟그룹, 보안그룹(사설만), ASG. DLC는 인증이 없으므로 **8080 공개 금지**(루프백 또는 VPC 내부만).

장점: Spot·단가 최적화, 사이드카/배치 잡 동거, 미들웨어 자유.  
단점: 헬스체크·롤링·모델 동기화·드라이버/AMI를 직접 운영.

### 비교와 선택 기준


|       | SageMaker Endpoint              | EC2 직접                   |
| ----- | ------------------------------- | ------------------------ |
| 공식 지원 | CDK + DLC 완비                    | DIY (`serve` / DLC 수동)   |
| 보안    | IAM Invoke, 네트워크 격리 쉬움          | SG/ALB/자체 인증 필요          |
| 스케일   | Endpoint autoscaling            | ASG + 타겟그룹               |
| 비용    | GPU on-demand (블로그 g5 \~$1.4/h) | 동일 GPU면 보통 더 저렴, Spot 가능 |
| 적합    | 프로덕션 API, Lambda/SQS 연동         | 고정 트래픽·커스텀 파이프라인·단가 최적화  |


**결론:** 에이전트/티켓 분류를 서비스로 붙이면 **SageMaker** `ml.g6.xlarge`**(또는** `ml.g5.xlarge`**) + E4B CDK**가 기본. EC2는 Spot 단가, 같은 호스트에 전처리·배치를 같이 두거나 SageMaker 용량/쿼터가 막힐 때의 대안.

## 커스터마이징·학습·연구 기록

- `skills/customize-decider/SKILL.md`: 에이전트용 스킬입니다. 결정 정의 → 평가셋 구성 → 공개 모델 비교 → 학습 여부 판단 → 학습·모니터링 → 보정 → 기존 모델과 비교까지 단계별로 안내합니다.
- **학습**: `training/recipe.sh all`(호스트 1대, GPU 1\~8장) 또는 `training/run_recipe.sh all`(분산 8 GPU + S3 저장). RTX 3090 한 장으로 약 11시간, H100 8장(`FAST=1`)으로 약 1시간 10분이 걸립니다.
- **벤치마크**: `benchmark/`로 서빙 변경 전후의 RPS, 지연, 100만 건당 비용을 비교할 수 있습니다.
- **연구 투명성**: v9부터 학습 전에 예측과 실패 조건을 미리 등록(preregistration)하고, 결과는 나중에 덧붙이기만 합니다. 기준에 못 미친 실험도 기록에 남깁니다.
- **왜 2B인가**: 가진 하드웨어로 서빙과 재학습이 모두 가능해 실험이 빠르고, 의미 있는 작업을 하기에 충분한 크기이기 때문입니다.

## 커뮤니티 사례


| 저장소                                                                                                       | 내용                                                        |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [Vivek0712/strands-decider-agentcore](https://github.com/Vivek0712/strands-decider-agentcore)             | AgentCore CLI로 Decider를 AgentCore Runtime에 배포합니다.         |
| [mikegc-aws/decider-gated-agentcore-memory](https://github.com/mikegc-aws/decider-gated-agentcore-memory) | AgentCore Memory 장기 검색 여부를 Decider가 판단하는 실험적 PoC입니다.      |
| [Akemid/strands-decider-patterns](https://github.com/Akemid/strands-decider-patterns)                     | 도구 선택·가드레일 패턴과 Bedrock 폴백 구성입니다.                          |
| [Hasalawa/decider-triage](https://github.com/Hasalawa/decider-triage)                                     | 라우팅·도구 안전성 판단은 Decider가, 텍스트 작성은 Bedrock이 맡는 지원 에이전트입니다.  |
| [royosherove/graphlin](https://github.com/royosherove/graphlin)                                           | JEV와 Strands Decider로 코딩 에이전트의 아키텍처·활동 다이어그램을 실시간으로 만듭니다. |
| [milan9527/StrandsDecider](https://github.com/milan9527/StrandsDecider)                                   | S3 + CloudFront OAC 프런트엔드와 ECS Fargate 백엔드로 구성한 데모입니다.    |


Agent Experience Hackathon 출품작 중 `#strands-decider-2b` 태그가 붙은 프로젝트가 많습니다.

## Amazon 사례 — Sonnet 4.6을 Decider로 교체 (98% 절감)

> 출처: [Amazon saves millions by replacing Sonnet 4.6 with Strands Decider](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/) · Strands Agents 블로그 · 2026-11-09  
> 작성자: Carr Onstott, Mehul Jaiswal (Amazon 내부 티켓 시스템 엔지니어), Strands Agents Team

### 요약

Amazon·AWS 내부 티켓 시스템에서 티켓 라벨링에 쓰던 **Claude Sonnet 4.6을 Strands Decider E4B(Gemma 4 기반)로 교체**했습니다. 결과적으로 **비용은 98% 줄고, 호출당 지연은 78% 줄었으며**, 별도 티켓 세트에서 운영 중인 Sonnet보다 더 좋은 결과를 냈습니다. 연간 수백만 달러의 토큰 비용을 절감할 것으로 봅니다.

### 문제와 접근

- 티켓(고객 에스컬레이션, 용량 요청, 자동 알람, 보안 이슈 등)을 **고정된 20개 카테고리**로 분류하는 작업입니다.
- 기존에는 티켓마다 긴 시스템 프롬프트로 Sonnet을 호출해 JSON과 추론을 생성했습니다.
- 결정 모델은 텍스트를 생성하지 않고 상태·질문·선택지를 받아 **선택지별 확률과 신뢰도**를 돌려줍니다. 상태 = 티켓, 선택지 = 라벨 카탈로그로 그대로 대응됩니다.

### 적용 과정

1. 처음에는 **Decider 2B**로 시작했습니다.
2. 테스트 세트에서 **E4B(Gemma 4)** 가 확실히 좋아 교체했습니다.
3. 라벨을 적용할 **신뢰도 임계값을 튜닝**했습니다.
4. 별도 티켓 세트에서 E4B가 운영 Sonnet을 앞섰습니다.

프롬프트 재작성도 필요했습니다.

- 운영용 라벨 설명만으로도 Decider의 **4,096토큰 창을 넘었기** 때문에, 20개 라벨을 각각 **1\~2문장의 기준**으로 다시 쓰고 `none` 옵션을 더해 **choice 질문 하나**로 보냈습니다.
- 티켓 데이터는 Sonnet과 같지만, 긴 티켓은 **가장 오래된 댓글부터 잘라** 창에 맞춥니다.

### 아키텍처

모든 구성 요소가 표준 AWS 서비스입니다(티켓 API 제외).

```
티켓 이벤트 → SQS → Lambda(라벨링)
                      ├─ AWS AppConfig : 3-way 스위치 (메시지마다 읽음)
                      ├─ SageMaker AI Endpoint : Strands Decider E4B
                      │    · ml.g5.xlarge (A10G 1장, 24GB), 모델 메모리 약 19GiB
                      │    · 요청당 약 640ms, 트래픽 기반 오토스케일
                      │    · 네트워크 격리(컨테이너 외부 통신 차단), 기동 시 S3에서 가중치 복사
                      └─ CloudWatch : 로그·지표·알람
IaC: CDK (Beta → Gamma → Prod)
```

### 3-way 스위치 (AWS AppConfig)


| 모드        | 동작                                                     |
| --------- | ------------------------------------------------------ |
| `BEDROCK` | Sonnet만 라벨링하고 Decider는 호출하지 않습니다.                      |
| `DUAL`    | 두 모델을 병렬 호출합니다. Sonnet 결과를 티켓에 적용하고 Decider 결과와 비교합니다. |
| `DECIDER` | Decider만 라벨링하고 Bedrock은 호출하지 않습니다.                     |


### 측정 결과


| 항목        | Claude Sonnet 4.6 | Strands Decider E4B | 개선  |
| --------- | ----------------- | ------------------- | --- |
| 호출당 평균 시간 | 4,215 ms          | 897 ms              | 78% |
| 티켓당 입력 토큰 | 6,554             | 2,009               |     |
| 티켓당 출력 토큰 | 170               | 1                   |     |
| 티켓당 총 토큰  | 6,724             | 2,010               | 70% |


- 토큰이 줄어든 이유는 ① 출력이 JSON 대신 선택 1개이고, ② 프롬프트가 훨씬 짧기 때문입니다.
- Sonnet을 끈 날 **Lambda 평균 실행 시간이 74% 감소**했습니다.

### 비용 (us-east-1 온디맨드 공시가)


|         | Sonnet 4.6 (Bedrock, global endpoint) | Decider E4B (SageMaker ml.g5.xlarge) |
| ------- | ------------------------------------- | ------------------------------------ |
| 단가      | 입력 $3.00 / 출력 $15.00 (100만 토큰당)       | 인스턴스 시간당 $1.408                      |
| 티켓당     | $0.0222                               | 약 $0.0005                            |
| 100만 건당 | 약 $22,200                             | 약 $500                               |


- 인스턴스 1대가 시간당 약 **5,600건**을 처리하므로 100% 가동 시 100만 건당 **$250**입니다.
- 버스트용 여유를 두어 실제로는 약 **$500/100만 건**이며, Sonnet 대비 **약 44배 저렴(98% 절감)** 합니다.
- Lambda 비용은 제외한 수치이며, Lambda 실행 시간도 줄었습니다.

### 운영 팁 — JSON 키 순서

같은 내용이라도 **JSON 키 순서가 다르면 출력 확률이 최대 0.19까지 달라졌습니다.** 현재는 모든 요청을 **키 정렬** 후 보냅니다. API 뒤에서 결정 모델을 서빙한다면 실행 결과를 비교하기 전에 반드시 확인해야 합니다.

### 교훈

**잘 맞은 조건**

- 알려진 카탈로그에서 고르는 **닫힌 선택(closed choice)** 작업입니다.
- **처리량이 많습니다.**
- 기존 경로의 **지연이 중요**했습니다.

**대체하기 어려운 경우**

- 라벨에 **설명 텍스트**가 필요한 경우입니다.
- 카탈로그가 **매주 바뀌는** 경우입니다.
- 유용한 맥락이 **수천 토큰 안에 들어가지 않는** 경우입니다.

**모델과 무관하게 따라 할 점**: 3-way 스위치입니다. 실제 트래픽에서 두 모델을 함께 돌리고 불일치 건을 판정한 것이 어떤 오프라인 평가보다 유용했으며, 롤백은 **45초짜리 설정 변경**으로 끝납니다.

### 읽을 때 유의할 점 (작성자 해석)

- 98% 절감은 **2B가 아니라 E4B(Gemma 4 기반)** 의 결과입니다.
- 지연 수치가 **640ms(인스턴스 처리)** 와 **897ms(호출당 평균)** 두 가지로 나옵니다. 차이는 네트워크·대기 시간으로 보이지만 글에는 설명이 없습니다.
- 정확도는 "Sonnet보다 낫다"고만 했고 **구체 수치는 공개하지 않았습니다.**
- 실제 티켓 처리량도 공개하지 않았습니다. 100만 건당 약 $21,700을 절감하므로, 연 수백만 달러가 되려면 **연간 수억 건** 규모로 추정됩니다.

## 직접 확인하지 못한 부분

- AWS 공식 문서(docs.aws.amazon.com)를 검색했지만 Strands Decider 항목은 찾지 못했습니다. 이 노트는 GitHub README, 배포 가이드, 예제 문서를 기준으로 작성했습니다.
- (2026-11-11 기준) 공식 발표 블로그 글은 찾지 못했습니다. 이후 2026-11-09자 Strands Agents 블로그의 Amazon 사례 글을 확인해 [Amazon 사례](#amazon-사례--sonnet-46을-decider로-교체-98-절감) 섹션에 반영했습니다.
- `strands-labs`는 정식 SDK 조직인 `strands-agents`와 별개인 Labs 조직입니다. 그래서 GA 제품이라기보다 실험적이고 연구 성격이 강한 프로젝트로 보입니다(작성자 해석).
- 성능 수치는 모두 저장소에서 스스로 측정한 값입니다.
- 한국어 입력 성능은 문서에 따로 나와 있지 않아 직접 평가가 필요합니다.

## 참고 링크

- GitHub: [https://github.com/strands-labs/strands-decider](https://github.com/strands-labs/strands-decider)
- Hugging Face: [https://huggingface.co/StrandsAgents](https://huggingface.co/StrandsAgents)
- Strands Agents SDK: [https://github.com/strands-agents/sdk-python](https://github.com/strands-agents/sdk-python)
- Strands 예제: [https://github.com/strands-labs/strands-decider/tree/main/examples/strands](https://github.com/strands-labs/strands-decider/tree/main/examples/strands)
- 배포 가이드: [https://github.com/strands-labs/strands-decider/tree/main/deploy](https://github.com/strands-labs/strands-decider/tree/main/deploy)
- SageMaker CDK: [https://github.com/strands-labs/strands-decider/blob/main/deploy/sagemaker/README.md](https://github.com/strands-labs/strands-decider/blob/main/deploy/sagemaker/README.md)
- strands-decider DLC: [https://aws.github.io/deep-learning-containers/strands-decider/](https://aws.github.io/deep-learning-containers/strands-decider/)
- Amazon 사례 블로그: [https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/)
