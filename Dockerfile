FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CAREBOT_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[openai]"
COPY evals ./evals

RUN useradd --create-home carebot && mkdir -p /data && chown carebot /data
USER carebot
VOLUME /data
# Only used in webhook mode (CAREBOT_WEBHOOK_URL set). Polling needs no open port.
EXPOSE 8080

CMD ["carebot", "run"]
