FROM python:3.11-slim

WORKDIR /app

ARG EMB_MODEL=sentence-transformers/all-mpnet-base-v2
ARG RELF1_EMB_MODEL=BAAI/bge-m3
ARG RELF1_EMB_STRATEGY=bge-m3

ENV EMB_MODEL=${EMB_MODEL} \
    RELF1_EMB_MODEL=${RELF1_EMB_MODEL} \
    RELF1_EMB_STRATEGY=${RELF1_EMB_STRATEGY} \
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

relf1_model_name = os.environ["RELF1_EMB_MODEL"]
relf1_strategy = os.environ.get("RELF1_EMB_STRATEGY", "auto").strip().lower()
if relf1_model_name and relf1_model_name != model_name:
    if relf1_strategy in {"bge-m3", "m3"} or relf1_model_name == "BAAI/bge-m3":
        BGEM3FlagModel(relf1_model_name, use_fp16=False)
        print(f"BGEM3FlagModel cached: {relf1_model_name}")
    elif relf1_strategy in {"bge", "bge-v1.5", "bge-v15", "bge-v1_5"} or relf1_model_name.startswith("BAAI/bge-"):
        FlagModel(relf1_model_name, use_fp16=False)
        print(f"FlagModel cached: {relf1_model_name}")
    else:
        SentenceTransformer(relf1_model_name, device="cpu")
        print(f"SentenceTransformer model cached: {relf1_model_name}")
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMB_LOCAL_FILES_ONLY=true

COPY . .

RUN chmod +x /app/start.sh

EXPOSE 9000

CMD ["/app/start.sh"]
