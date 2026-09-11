PROMPT_TPCH_INTERNAL_KNOWLEDGE_TEMPLATE = """
Answer the QUESTION using only your internal knowledge of the standard TPC-H
benchmark. You are NOT given rows from the database instance.

This mode is intentionally different from a context-grounded or retrieval-based
mode. Accuracy is more important than coverage. Returning [] is better than
guessing.

TPCH LOGICAL SCHEMA:
- region(regionkey, name, comment)
- nation(nationkey, name, regionkey, comment)
- supplier(suppkey, name, address, nationkey, phone, acctbal, comment)
- customer(custkey, name, address, nationkey, phone, acctbal, mktsegment, comment)
- orders(orderkey, custkey, orderstatus, totalprice, orderdate, orderpriority, clerk, shippriority, comment)
- lineitem(orderkey, partkey, suppkey, linenumber, quantity, extendedprice, discount, tax, returnflag, linestatus, shipdate, commitdate, receiptdate, comment)
- part(partkey, name, mfgr, brand, type, size, container, retailprice, comment)
- partsupp(partkey, suppkey, availqty, supplycost, comment)

PRIMARY KEYS:
- region: regionkey
- nation: nationkey
- supplier: suppkey
- customer: custkey
- orders: orderkey
- lineitem: orderkey, linenumber
- part: partkey
- partsupp: partkey, suppkey

INTERNAL-KNOWLEDGE LIMITS:
- You may use stable public knowledge about the TPC-H schema and standard domain
  conventions.
- You may use standard TPC-H key facts only when you know them exactly.
- Do NOT invent generated instance data, such as customer rows, supplier rows,
  order rows, lineitem rows, comments, prices, quantities, dates, phone numbers,
  account balances, or local row numbers.
- Do NOT assume access to this project's CSV files.
- Do NOT use local row identifiers such as "region_1" or fields such as
  "region_rownum". Those are local instance metadata and are not part of your
  internal TPC-H knowledge.
- If the exact answer or exact provenance cannot be known from internal knowledge,
  return [].

PROVENANCE IDENTIFIERS:
- Provenance identifiers in this benchmark mode are SEMANTIC TPC-H identifiers
  based on primary-key values, not local CSV row numbers.
- Format each identifier as "<table_name>_<primary_key_value>".
- Examples:
  - The region tuple with regionkey = 0 is "region_0".
  - The nation tuple with nationkey = 3 is "nation_3".
  - The part tuple with partkey = 42 is "part_42".
  - The lineitem tuple with orderkey = 100 and linenumber = 2 is
    "lineitem_100_2".
  - The partsupp tuple with partkey = 42 and suppkey = 7 is "partsupp_42_7".

PROVENANCE RULES:
- The provenance field MUST be a list of lists of provenance identifiers.
- Each inner list is one sufficient set of source tuples that produces the
  result tuple.
- For a single-table result, use one inner list containing the source tuple.
- For a join result, use one inner list containing all joined source tuples.
- For alternative derivations of the same result, use multiple inner lists.
- For aggregation results, include the complete set of contributing source
  tuples only if you know it exactly. Otherwise return [].

OUTPUT RULES:
- Return ONLY valid JSON and no introductory text.
- The entire output MUST be a JSON array.
- Each array element MUST be an object with EXACTLY these keys:
  - result: an object representing one output tuple
  - provenance: a Why[X] provenance expression for that tuple
- Use logical TPC-H column names in result objects, such as "regionkey", "name",
  "nationkey", "orderkey", and "totalprice".
- For computed values, use clear result keys such as "count", "total",
  "sum_quantity", "avg_price", "min_orderdate", or "max_orderdate".
- Do NOT output SQL.
- Do NOT output explanations, comments, markdown, or code fences.
- Do NOT add extra keys.
- If there are no results, return [].

JSON SCHEMA:
[{{"result": {{...}}, "provenance": [["t1", "t2"], ["t3"]]}}]

QUESTION:
{question}
"""


PROMPT_RELF_INTERNAL_KNOWLEDGE_TEMPLATE = """
Answer the QUESTION using your internal knowledge about Formula 1.

Return only a valid JSON array. Each object must contain:
- "id": an integer starting at 1 and increasing by 1 for each row.
- The required output columns listed below.

Answer the question directly. Do not invent facts or unknown values.
If you cannot answer, return [].
Do not include explanations, markdown, or additional fields.

REQUIRED OUTPUT COLUMNS:
{output_columns}

QUESTION:
{question}
"""


PROMPT_RELARXIV_INTERNAL_KNOWLEDGE_TEMPLATE = """
Answer the QUESTION using your internal knowledge about arXiv papers
and their authors.

Return only a valid JSON array. Each object must contain:
- "id": an integer starting at 1 and increasing by 1 for each row.
- The required output columns listed below.

Answer the question directly. Do not invent facts or unknown values.
If you cannot answer, return [].
Do not include explanations, markdown, or additional fields.

REQUIRED OUTPUT COLUMNS:
{output_columns}

QUESTION:
{question}
"""


PROMPT_RELF1_MINIMAL_INTERNAL_KNOWLEDGE_TEMPLATE = PROMPT_RELF_INTERNAL_KNOWLEDGE_TEMPLATE
PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE = PROMPT_TPCH_INTERNAL_KNOWLEDGE_TEMPLATE


def normalize_internal_result_id(row: dict, index: int) -> dict:
    """Fill a missing synthetic ID using the zero-based response position."""
    row = dict(row)
    if row.get("id") is None:
        row["id"] = index + 1
    if type(row["id"]) is not int or row["id"] != index + 1:
        raise ValueError(f"Item {index} must have sequential integer id {index + 1}")
    return row


def uses_plain_internal_results(domain: str) -> bool:
    return domain.strip().lower() in {
        "relf", "relf1", "rel-f1", "f1", "formula1", "formula-1",
        "rel_arxiv", "rel-arxiv", "relarxiv",
    }


def get_internal_knowledge_prompt_template(domain: str) -> str:
    normalized = domain.strip().lower()
    if normalized in {"relf", "relf1", "rel-f1", "f1", "formula1", "formula-1"}:
        return PROMPT_RELF_INTERNAL_KNOWLEDGE_TEMPLATE
    if normalized in {"tpch", "tpc-h"}:
        return PROMPT_TPCH_INTERNAL_KNOWLEDGE_TEMPLATE
    if normalized in {"rel_arxiv", "rel-arxiv", "relarxiv"}:
        return PROMPT_RELARXIV_INTERNAL_KNOWLEDGE_TEMPLATE
    raise ValueError(f"Unsupported internal-knowledge prompt domain: {domain}")


def build_internal_knowledge_prompt(
    domain: str, question: str, output_columns: list[str] | None = None,
) -> str:
    columns = list(dict.fromkeys(column for column in output_columns or [] if column != "id"))
    template = get_internal_knowledge_prompt_template(domain)
    # An explicit empty projection requests rows without column instructions.
    if output_columns == []:
        template = template.replace("- The required output columns listed below.\n", "")
        template = template.replace("REQUIRED OUTPUT COLUMNS:\n{output_columns}\n\n", "")
        template = template.replace("Do not include explanations, markdown, or additional fields.",
                                    "Do not include explanations or markdown.")
    return template.format(
        question=question,
        output_columns=(", ".join(columns) if columns else
                        "Use column names that directly match the information requested in the QUESTION."),
    )
