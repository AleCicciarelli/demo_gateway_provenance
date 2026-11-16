"""Context/output budgeting for source-backed RAG, without dropping evidence."""
from functools import lru_cache
import json
import os


@lru_cache(maxsize=4)
def _tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def count_tokens(text):
    path = os.getenv('RAG_TOKENIZER_PATH', '').strip()
    if path:
        return len(_tokenizer(path).encode(text, add_special_tokens=False))
    # Conservative byte-based estimate, not a model-specific token count.
    return len(text.encode('utf-8'))


def context_window(provider):
    if provider.strip().lower() == 'ollama':
        return int(os.getenv('OLLAMA_NUM_CTX', '8192'))
    return int(os.getenv('RAG_CONTEXT_TOKENS') or os.getenv('MATCH_CONTEXT_TOKENS')
               or os.getenv('OLLAMA_NUM_CTX', '8192'))


def retrieval_limits(provider, max_output_tokens, question):
    """Bound discovery by usable model context, with explicit search backstops."""
    window = context_window(provider)
    windows = int(os.getenv('RETRIEVER_CONTEXT_WINDOWS', '1'))
    patience = int(os.getenv('RETRIEVER_NO_NEW_ROWS_PATIENCE', '2'))
    rounds = int(os.getenv('RETRIEVER_MAX_ROUNDS', '10'))
    reserve = int(os.getenv('RAG_PROMPT_RESERVE_TOKENS', '1024'))
    usable = window - min(max_output_tokens, max(1, window // 4)) - reserve
    # Reserve instructions/bindings in addition to the retrieval query itself.
    budget = (usable - count_tokens(question) - 1024) * windows
    if windows <= 0 or patience <= 0 or rounds <= 0 or reserve < 0 or budget <= 0:
        raise ValueError('Retrieval context budget, window count, patience and round cap must be positive')
    return budget, patience, rounds


def plan_batches(task, context, make_prompt, *, context_tokens, max_output_tokens):
    """Split only to fit input context, retaining all target/support pairs."""
    reserve = int(os.getenv('RAG_PROMPT_RESERVE_TOKENS', '1024'))
    if context_tokens <= 0 or max_output_tokens <= 0 or reserve < 0:
        raise ValueError('RAG context/output budgets must be positive; reserve must be nonnegative')
    # Keep room for input even when the global output limit equals the window.
    output_limit = min(max_output_tokens, max(1, context_tokens // 4))
    prompt_limit = context_tokens - output_limit - reserve
    if prompt_limit <= 0:
        raise ValueError('RAG context window leaves no prompt budget after output/retry reserves')
    table = task['table_name']
    targets = list((context.get(table) or {}).items())
    support = [(t, rid, row) for t, rows in context.items() if t != table for rid, row in rows.items()]
    required = set()
    for key in ('columns', 'select_columns', 'join_keys', 'group_by_columns', 'aggregate_columns'):
        required.update(str(c).lower() for c in task.get(key) or [] if c != '*')

    def build(target_rows, support_rows):
        ctx = {table: dict(target_rows)}
        for t, rid, row in support_rows:
            ctx.setdefault(t, {})[rid] = row
        return ctx

    def partition(target_rows, support_rows):
        ctx = build(target_rows, support_rows)
        prompt = make_prompt(ctx)
        expected = [{'row_id': rid, 'values': row if task.get('all_columns') else
                     {k: v for k, v in row.items() if k.lower() in required}}
                    for rid, row in target_rows]
        prompt_size = count_tokens(prompt)
        # Allow formatting overhead beyond the compact JSON estimate.
        output_size = count_tokens(json.dumps(expected, ensure_ascii=False)) + 128
        if prompt_size <= prompt_limit:
            return [{'context': ctx, 'prompt': prompt, 'prompt_tokens': prompt_size,
                     'estimated_output_tokens': output_size, 'max_output_tokens': output_limit,
                     'context_tokens': context_tokens}]
        if len(target_rows) > 1:
            mid = len(target_rows) // 2
            return partition(target_rows[:mid], support_rows) + partition(target_rows[mid:], support_rows)
        if len(support_rows) > 1:
            mid = len(support_rows) // 2
            return partition(target_rows, support_rows[:mid]) + partition(target_rows, support_rows[mid:])
        raise ValueError('A single RAG row/prompt exceeds the configured input context budget; '
                         'increase the supported context/output limits or reduce requested columns')

    return partition(targets, support)
