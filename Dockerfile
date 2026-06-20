FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN sed -i -e 's#http://deb.debian.org/debian-security#http://mirrors.aliyun.com/debian-security#g' -e 's#http://deb.debian.org/debian#http://mirrors.aliyun.com/debian#g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl \
        libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -v "^torch==" requirements.lock.txt > requirements.lock.no-torch.txt \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --upgrade pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --no-deps -r requirements.lock.no-torch.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.7.1+cpu

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
