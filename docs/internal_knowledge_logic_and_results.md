# Internal-Knowledge Logic and Results

This note documents the internal-knowledge evaluation mode and summarizes the results currently available in `evaluation/`.


## Main Files

- `prompt_internal_knowledge.py`: prompt template for internal-knowledge answering.
- `run_internal_knowledge_eval.py`: runs root-level or leaf-level natural-language questions through the internal-knowledge prompt.
- `evaluate_internal_knowledge_outputs.py`: evaluates answer and provenance sets against ground truth.
- `run_oar_internal_knowledge_eval.sh`: OAR wrapper that starts Ollama, runs the evaluation, and writes metrics/plots.
- `evaluation/ground_truth_queries.json`: root query ground truth.
- `evaluation/ground_truth_leaf_tasks.json`: leaf-task ground truth.

## Runtime pipeline

`run_internal_knowledge_eval.py` loads evaluation questions and converts them into records. In root mode, each record is a full natural-language question from `evaluation/questions.json`. In leaf mode, each record is one leaf natural-language question from `leaf_tasks`.

For each record, the runner builds a prompt using `PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE`. The prompt includes:

- the logical TPC-H schema,
- primary-key definitions,
- rules forbidding access to local CSV rows,
- rules forbidding invented generated instance data,
- semantic provenance identifier format,
- the required JSON output schema.

The model must return a JSON array of objects with exactly:

```json
[
  {
    "result": {},
    "provenance": [["table_primarykey"]]
  }
]
```

The runner parses and validates the JSON shape. A record is marked `ok=true` only if the model output is valid according to this schema.

## Provenance Identifiers (to review)

Internal-knowledge provenance uses semantic TPC-H identifiers based on primary keys, not local CSV row numbers.

Examples:

- `region_0` for `regionkey = 0`
- `nation_3` for `nationkey = 3`
- `part_42` for `partkey = 42`
- `lineitem_100_2` for `orderkey = 100, linenumber = 2`
- `partsupp_42_7` for `partkey = 42, suppkey = 7`

The evaluator maps local row numbers from the ground truth to these semantic identifiers when needed.

This distinction is important: raw row ids in the ground-truth files should not be compared directly against internal-knowledge predictions. The ground truth is produced from the local CSV instance and often uses local row-number ids such as `customer_123` or `orders_456`. Those ids are implementation details of this project dataset. The internal-knowledge prompt explicitly forbids using them because a model without CSV access cannot know local row numbers.

For internal-knowledge evaluation, provenance comparison is only meaningful after converting local row-number ids into semantic primary-key ids. For example, if local row `customer_123` has `c_custkey = 4567`, the comparable semantic id is `customer_4567`. The evaluator performs this conversion using the CSV files and the primary-key definitions.

Even after this conversion, most semantic ids are still not knowable from internal model knowledge, because the generated TPC-H instance values are not provided in the prompt. The conversion only makes the identifier formats comparable; it does not make the task information-complete.

## Evaluation Modes

Root mode compares complete query answers and complete query provenance. It evaluates:

- answer exact match,
- answer precision/recall/F1,
- provenance exact match,
- provenance precision/recall/F1.

Leaf mode compares predicted provenance identifiers for each leaf task after converting ground-truth local row ids to semantic primary-key ids. It is closer to the planner-first leaf evaluation, but without retrieval context.


### Root Internal-Knowledge Runs

The two completed root-level runs currently have identical summary metrics.

| Run | Records | OK rate | Valid output rate | Answer exact match | Answer micro F1 | Provenance exact match | Provenance micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `internal_knowledge_metrics_124896` | 46 | 0.8913 | 0.8913 | 0.0435 | 0.009479 | 0.0435 | 0.030864 |
| `internal_knowledge_metrics_124900` | 46 | 0.8913 | 0.8913 | 0.0435 | 0.009479 | 0.0435 | 0.030864 |

For both runs:

- answer TP = 1, FP = 29, FN = 180,
- provenance TP = 5, FP = 42, FN = 272.

### Leaf Internal-Knowledge Runs

| Run | Model | Records | OK rate | Valid output rate | TP | FP | FN | Micro row F1 | Macro row F1 | Hallucination-free rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `internal_knowledge_leaf_metrics_8b_125040` | 8B | 80 | 0.7250 | 0.7250 | 20 | 15 | 5877220 | 0.00000681 | 0.000574 | 0.8125 |
| `internal_knowledge_leaf_metrics_70b_125049` | 70B | 80 | 1.0000 | 1.0000 | 30 | 0 | 5877210 | 0.00001021 | 0.075000 | 1.0000 |

The 70B leaf run is cleaner than the 8B leaf run: all outputs are valid, no hallucinated row identifiers are reported, and it recovers slightly more true positives. However, recall is still extremely low because the task asks for exact database-instance rows without giving the model access to the database instance.

## Discussion

The root internal-knowledge results confirm that a language model cannot reliably reconstruct exact TPC-H instance answers from schema knowledge alone. The model often produces valid JSON, but valid formatting is not enough. The answer and provenance sets rarely match the project-specific ground truth.

This is expected. TPC-H defines a schema and a data generator, but the exact generated rows, row values, local row numbers, primary-key values appearing in each answer, and most query answers are instance-specific. Without retrieval or database execution, the model does not have enough information to recover those tuples. The prompt explicitly instructs the model not to invent generated instance data, so the best conservative behavior is often to return `[]`.

The root-level answer micro F1 is only `0.009479`, and the provenance micro F1 is `0.030864`. This indicates that the model sometimes knows stable schema-level facts or simple benchmark facts, but it cannot cover the actual answer sets. Exact match is also low: only `4.35%` of root queries exactly match for both answers and provenance.

The leaf-level comparison is useful because it isolates row/provenance recall. The 70B model performs better than 8B in this setting:

- higher valid output rate: `1.0000` vs `0.7250`,
- fewer hallucinations: `0` vs `15`,
- higher TP count: `30` vs `20`,
- higher macro row F1: `0.075000` vs `0.000574`.

Still, both models miss almost all expected rows. The 70B model has only 30 true positives against 5,877,240 expected semantic identifiers. This is not a failure of JSON formatting; it is an information-access limitation.

