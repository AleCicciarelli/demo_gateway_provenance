FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
from transformers import TrainingArguments
print("TrainingArguments OK")

from FlagEmbedding import BGEM3FlagModel, FlagModel
print("FlagEmbedding OK")
PY

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 9000

CMD ["/app/start.sh"]