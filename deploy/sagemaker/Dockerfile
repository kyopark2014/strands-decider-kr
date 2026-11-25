# SageMaker inference image for strands-decider v2 (PyPI package).
# Built by `cdk deploy -c image=build`. No local src/ required — installs
# strands-decider from PyPI, which loads decision_config.json (v2) checkpoints.
# DEVICE=cuda (default) for GPU; DEVICE=cpu omits flash-linear-attention.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /opt/strands-decider

ARG DEVICE=cuda
ARG STRANDS_DECIDER_VERSION=0.4.0
# transformers pin: last release tested with this image's torch 2.7.1
ARG TRANSFORMERS_VERSION=5.19.0

RUN pip install "strands-decider[${DEVICE}]==${STRANDS_DECIDER_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

# Prebuilt causal-conv1d wheel for CUDA 12 / torch 2.7 / Python 3.11
ARG CAUSAL_CONV1D_VERSION=1.7.0
RUN if [ "${DEVICE}" = cuda ]; then \
      abi=$(python -c "import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())") && \
      pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CAUSAL_CONV1D_VERSION}/causal_conv1d-${CAUSAL_CONV1D_VERSION}+cu12torch2.7cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"; \
    fi

ENV DECIDER_CHECKPOINT=/opt/ml/model/checkpoint \
    DECIDER_DEVICE=${DEVICE} \
    DECIDER_ARGS="" \
    HF_HUB_OFFLINE=1

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "exec strands-decider serve \"$DECIDER_CHECKPOINT\" --host 0.0.0.0 --port 8080 --device \"$DECIDER_DEVICE\" $DECIDER_ARGS"]
