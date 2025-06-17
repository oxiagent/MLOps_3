FROM bitnami/pytorch:2.3.1

RUN mkdir -p /app/.cache/wandb
ENV WANDB_CACHE_DIR=/app/.cache/wandb

RUN mkdir -p /app/model

WORKDIR /app

COPY requirements.txt requirements.txt
COPY download_model.py download_model.py
COPY main.py main.py
COPY start.sh start.sh

RUN pip install -r requirements.txt

ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY

# ❗️ Додаємо команду для завантаження моделі
RUN python download_model.py

RUN rm /opt/bitnami/scripts/pytorch/entrypoint.sh

ENTRYPOINT ["./start.sh"]
