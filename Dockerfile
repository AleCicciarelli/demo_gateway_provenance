FROM python:3.11-slim

WORKDIR /app

ARG EMB_MODEL=sentence-transformers/all-mpnet-base-v2

ENV EMB_MODEL=${EMB_MODEL} \
    HF_HOME=/opt/huggingface \
    TRANSFORMERS_CACHE=/opt/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence-transformers

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import os

from transformers import TrainingArguments
print("TrainingArguments OK")

from FlagEmbedding import BGEM3FlagModel, FlagModel
print("FlagEmbedding OK")

from sentence_transformers import SentenceTransformer
model_name = os.environ["EMB_MODEL"]
SentenceTransformer(model_name, device="cpu")
print(f"SentenceTransformer model cached: {model_name}")
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMB_LOCAL_FILES_ONLY=true

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 9000

CMD ["/app/start.sh"]
