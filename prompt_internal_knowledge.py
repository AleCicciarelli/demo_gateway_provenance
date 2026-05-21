PROMPT_INTERNAL_KNOWLEDGE_TEMPLATE = """
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
