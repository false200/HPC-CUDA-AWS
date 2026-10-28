FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev ca-certificates curl unzip \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip /var/lib/apt/lists/*

WORKDIR /opt
COPY src/engine.c /opt/src/engine.c
COPY kernels/mc_risk.cu /opt/kernels/mc_risk.cu
RUN gcc -O3 -std=c11 -o /opt/mc_engine /opt/src/engine.c -pthread -ldl -lm

COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENV MC_KERNEL=/opt/kernels/mc_risk.cu \
    NVRTC_PATH=/usr/local/cuda/lib64/libnvrtc.so.12 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64

ENTRYPOINT ["/opt/entrypoint.sh"]
