from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set
import json

import sqlglot
from sqlglot import exp


AGGREGATE_TYPES = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


# =========================
# Dataclasses
# =========================
''' 
Represents a reference to a column, with its raw SQL, table alias, resolved table name, and column name.
frozen=True makes the dataclass immutable.
'''
@dataclass(frozen=True)
class ColumnRef:
    raw: str
    table_alias: Optional[str]
    table_name: Optional[str]
    column_name: str

'''
Represents a predicate in the WHERE clause, including its SQL, involved tables, columns, and whether it can be pushed down to leaf scans (single table scans).
'''

@dataclass
class PredicateSpec:
    sql: str
    tables: List[str]
    columns: List[ColumnRef]
    pushable: bool

'''
Represents a JOIN specification, including the type of join, the table being joined, its alias, the ON condition in SQL, and the columns involved in the ON condition.
'''
@dataclass
class JoinSpec:
    join_type: str
    table: str
    alias: Optional[str]
    on_sql: Optional[str]
    on_columns: List[ColumnRef]

'''
Represents an item in the SELECT clause, including its SQL, alias, whether it contains an aggregate function, and the columns involved.
'''

@dataclass
class SelectItem:
    sql: str
    alias: Optional[str]
    is_aggregate: bool
    columns: List[ColumnRef]

'''
Represents an item in the ORDER BY clause, including its SQL, the expression being ordered, whether it's descending, and the columns involved.
'''

@dataclass
class OrderByItem:
    sql: str
    expression: str
    desc: bool
    columns: List[ColumnRef]

'''
Represents an expression in the GROUP BY clause, including its SQL and the columns involved.
'''

@dataclass
class GroupByItem:
    sql: str
    columns: List[ColumnRef]

'''
Represents an aggregate function used in the SELECT clause, including the function name and its SQL representation.
'''

@dataclass
class AggregateSpec:
    func: str
    sql: str

'''
Represents a task for extracting data from a leaf node in the query plan (translatable in a single table scan).
Includes the table name, alias, type of scan operation, columns involved, local predicates that can be applied at the leaf level, join keys for joining with other tables, select columns needed for the final output, group by columns if applicable, and aggregate columns if applicable.'''

@dataclass
class LeafExtractionTask:
    table_name: str
    alias: Optional[str]
    scan_op: str
    columns: List[str]
    local_predicates: List[str]
    join_keys: List[str]
    select_columns: List[str]
    group_by_columns: List[str]
    aggregate_columns: List[str]

'''
Represents an operation to be applied after leaf extraction, such as grouping, aggregation, projection, ordering, or limiting.
Includes the type of operation and a payload with relevant details (e.g., group by keys, aggregate specifications, select expressions, order by items, limit value).
'''
@dataclass
class PostOp:
    op: str
    payload: Dict[str, Any]

'''
Represents the overall query plan, including the type of query (e.g., SELECT), the original SQL, a list of leaf extraction tasks, join specifications, and post-operations to be applied after leaf extraction.
'''

@dataclass
class QueryPlan:
    query_type: str
    sql: str
    leaf_tasks: List[LeafExtractionTask]
    joins: List[JoinSpec]
    post_ops: List[PostOp]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================
# AST helpers: to navigate the sqlglot AST and extract relevant components for planning
# =========================
'''
Used to flatten nested AND conditions in the WHERE or ON clause into a list of individual predicates.
'''
def _flatten_and_conditions(node: Optional[exp.Expression]) -> List[exp.Expression]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        print(f"Flattening AND condition: {node.sql()}")
        return _flatten_and_conditions(node.left) + _flatten_and_conditions(node.right)
    return [node]

'''
Checks if the given expression node or any of its subnodes contains an aggregate function (COUNT, SUM, AVG, MIN, MAX).

'''

def _contains_aggregate(node: exp.Expression) -> bool:
    return isinstance(node, AGGREGATE_TYPES) or any(
        isinstance(subnode, AGGREGATE_TYPES) for subnode in node.walk()
    )

'''
Extract the table name and the table alias from a sqlglot Table expression.
'''
def _extract_table_name(table_expr: exp.Table) -> str:
    return table_expr.name
def _extract_table_alias(table_expr: exp.Table) -> Optional[str]:
    return table_expr.alias_or_name if table_expr.alias else None

'''
Used to build a mapping from table aliases to their actual table names, based on the FROM clause and JOIN clauses in the SELECT statement.
'''

def _build_alias_map(tree: exp.Select) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}

    from_expr = tree.args.get("from_") or tree.args.get("from")
    if from_expr is not None:
        if getattr(from_expr, "this", None) is not None and isinstance(from_expr.this, exp.Table):
            t = from_expr.this
            name = _extract_table_name(t)
            alias = _extract_table_alias(t)
            alias_map[name] = name
            if alias:
                alias_map[alias] = name

    for join in tree.find_all(exp.Join):
        if isinstance(join.this, exp.Table):
            name = _extract_table_name(join.this)
            alias = _extract_table_alias(join.this)
            alias_map[name] = name
            if alias:
                alias_map[alias] = name
    print(f"Alias map: {alias_map}")
    return alias_map

'''
Used to normalize a column reference by resolving its table alias to the actual table name using the alias map, and creating a ColumnRef object that includes the raw SQL, table alias, resolved table name, and column name.
'''
def _normalize_column(col: exp.Column, alias_map: Dict[str, str]) -> ColumnRef:
    table_alias = col.table or None
    column_name = col.name
    table_name = alias_map.get(table_alias, table_alias) if table_alias else None
    return ColumnRef(
        raw=col.sql(),
        table_alias=table_alias,
        table_name=table_name,
        column_name=column_name,
    )
''' 
Used to normalize predicates
'''
def _normalize_predicate_sql(pred: exp.Expression, alias_map: Dict[str, str]) -> str:
    normalized = pred.copy()

    for col in list(normalized.find_all(exp.Column)):
        col.set("table", None)

    return normalized.sql()
'''
Extracts the base tables from the FROM clause of the SELECT statement, including their names, aliases, and raw SQL.
'''

def _extract_base_tables(tree: exp.Select) -> List[Dict[str, Any]]:
    tables = []
    from_expr = tree.args.get("from_") or tree.args.get("from")
    if from_expr is None:
        return tables

    candidates = []
    if getattr(from_expr, "this", None) is not None:
        candidates.append(from_expr.this)
    candidates.extend(list(from_expr.find_all(exp.Table)))

    seen = set()
    for node in candidates:
        if isinstance(node, exp.Table):
            name = _extract_table_name(node)
            alias = _extract_table_alias(node)
            key = (name, alias)
            if key not in seen:
                seen.add(key)
                tables.append({
                    "name": name,
                    "alias": alias,
                    "sql": node.sql(),
                })
    return tables

'''
Extracts the JOIN specifications from the SELECT statement, including the type of join, the table being joined, its alias, the ON condition in SQL, and the columns involved in the ON condition. 
Uses the alias map to resolve table aliases in the ON conditions.
'''

def _extract_joins(tree: exp.Select, alias_map: Dict[str, str]) -> List[JoinSpec]:
    joins: List[JoinSpec] = []

    for join in tree.find_all(exp.Join):
        if not isinstance(join.this, exp.Table):
            continue

        join_table = _extract_table_name(join.this)
        join_alias = _extract_table_alias(join.this)

        on_expr = join.args.get("on")
        on_conditions = _flatten_and_conditions(on_expr)
        print(f"Extracting JOIN: {join.sql()}")
        print(f"ON conditions: {[c.sql() for c in on_conditions]}")
        on_columns = [
            _normalize_column(col, alias_map)
            for condition in on_conditions
            for col in condition.find_all(exp.Column)
        ]

        joins.append(
            JoinSpec(
                join_type=join.args.get("kind", "JOIN"),
                table=join_table,
                alias=join_alias,
                on_sql=on_expr.sql() if on_expr else None,
                on_columns=on_columns,
            )
        )

    return joins

'''
Extracts all the items in the SELECT clause, including their SQL, aliases, whether they contain aggregate functions, and the columns involved.
'''
def _extract_select_items(tree: exp.Select, alias_map: Dict[str, str]) -> List[SelectItem]:
    out: List[SelectItem] = []

    for item in tree.expressions:
        alias = item.alias_or_name if item.alias else None
        cols = [_normalize_column(col, alias_map) for col in item.find_all(exp.Column)]
        out.append(
            SelectItem(
                sql=item.sql(),
                alias=alias,
                is_aggregate=_contains_aggregate(item),
                columns=cols,
            )
        )
    return out

'''
Extracts the GROUP BY items from the SELECT statement, including their SQL and the columns involved..
'''
def _extract_group_by(tree: exp.Select, alias_map: Dict[str, str]) -> List[GroupByItem]:
    group = tree.args.get("group")
    if group is None:
        return []

    out: List[GroupByItem] = []
    for expr_node in group.expressions:
        cols = [_normalize_column(col, alias_map) for col in expr_node.find_all(exp.Column)]
        out.append(GroupByItem(sql=expr_node.sql(), columns=cols))
    return out

'''
Extracts the ORDER BY items from the SELECT statement, including their SQL, the expression being ordered, whether it's descending, and the columns involved.
'''
def _extract_order_by(tree: exp.Select, alias_map: Dict[str, str]) -> List[OrderByItem]:
    order = tree.args.get("order")
    if order is None:
        return []

    out: List[OrderByItem] = []
    for ordered in order.expressions:
        expr_node = ordered.this if getattr(ordered, "this", None) is not None else ordered
        cols = [_normalize_column(col, alias_map) for col in expr_node.find_all(exp.Column)]
        out.append(
            OrderByItem(
                sql=ordered.sql(),
                expression=expr_node.sql(),
                desc=bool(getattr(ordered, "args", {}).get("desc")),
                columns=cols,
            )
        )
    return out

'''
Extract the limit value from the SELECT statement, if present. Tries to parse it as an integer, and returns None if it cannot be parsed.
'''
def _extract_limit(tree: exp.Select) -> Optional[int]:
    limit = tree.args.get("limit")
    if limit is None or limit.expression is None:
        return None

    try:
        return int(limit.expression.name)
    except Exception:
        try:
            return int(limit.expression.sql())
        except Exception:
            return None

'''
Extracts the aggregates functions used in the SELECT clause, including their function names and SQL representations.
'''
def _extract_aggregates(tree: exp.Select) -> List[AggregateSpec]:
    out: List[AggregateSpec] = []
    for item in tree.expressions:
        for node in item.walk():
            if isinstance(node, exp.Count):
                out.append(AggregateSpec(func="COUNT", sql=node.sql()))
            elif isinstance(node, exp.Sum):
                out.append(AggregateSpec(func="SUM", sql=node.sql()))
            elif isinstance(node, exp.Avg):
                out.append(AggregateSpec(func="AVG", sql=node.sql()))
            elif isinstance(node, exp.Min):
                out.append(AggregateSpec(func="MIN", sql=node.sql()))
            elif isinstance(node, exp.Max):
                out.append(AggregateSpec(func="MAX", sql=node.sql()))
    return out

'''
Extracts the predicates in the WHERE clause, including their SQL, involved tables, columns, and whether they can be pushed down to leaf scans (single table scans without OR).
'''
def _extract_predicates(tree: exp.Select, alias_map: Dict[str, str]) -> List[PredicateSpec]:
    where = tree.args.get("where")
    if where is None or where.this is None:
        return []

    preds = _flatten_and_conditions(where.this)
    out: List[PredicateSpec] = []

    for pred in preds:
        cols = [_normalize_column(col, alias_map) for col in pred.find_all(exp.Column)]
        tables = sorted({c.table_name for c in cols if c.table_name is not None})

        sql_upper = pred.sql().upper()
        pushable = len(tables) == 1

        out.append(
            PredicateSpec(
                sql=pred.sql(),
                tables=tables,
                columns=cols,
                pushable=pushable,
            )
        )
    return out


# =========================
# Planner helpers
# =========================

def _add_unique_str(target: List[str], value: Optional[str]) -> None:
    if value is not None and value not in target:
        target.append(value)

'''
Ensures that there is a LeafExtractionTask for the given table name in the leaf map, creating one if it doesn't exist. 
This is used to prepare the leaf tasks for each table involved in the query.
'''
def _ensure_leaf(
    leaf_map: Dict[str, LeafExtractionTask],
    table_name: str,
    alias: Optional[str],
) -> None:
    if table_name not in leaf_map:
        leaf_map[table_name] = LeafExtractionTask(
            table_name=table_name,
            alias=alias,
            scan_op="LeafScan",
            columns=[],
            local_predicates=[],
            join_keys=[],
            select_columns=[],
            group_by_columns=[],
            aggregate_columns=[],
        )

'''
If a task is a leaf scan (i.e., it only involves one table and does not contain OR), it updates the corresponding LeafExtractionTask 
in the leaf map to use a "FilterLeafScan" operation if it involves predicates or to a "LeafScan" operation otherwise.
'''
def _finalize_leaf_tasks(leaf_map: Dict[str, LeafExtractionTask]) -> List[LeafExtractionTask]:
    tasks = []
    for task in leaf_map.values():
        if task.local_predicates:
            task.scan_op = "FilterLeafScan"
        else:
            task.scan_op = "LeafScan"
        tasks.append(task)
    return tasks


# =========================
# Main planner
# =========================

def build_query_plan(sql: str) -> QueryPlan:
    # Parse the SQL query into a sqlglot AST
    tree = sqlglot.parse_one(sql)

    if not isinstance(tree, exp.Select):
        raise ValueError(f"Only select for now: {type(tree)}")

    #Extract all the components
    alias_map = _build_alias_map(tree)
    base_tables = _extract_base_tables(tree)
    joins = _extract_joins(tree, alias_map)
    select_items = _extract_select_items(tree, alias_map)
    group_by = _extract_group_by(tree, alias_map)
    order_by = _extract_order_by(tree, alias_map)
    predicates = _extract_predicates(tree, alias_map)
    aggregates = _extract_aggregates(tree)
    limit = _extract_limit(tree)

    #Build leaf tasks
    leaf_map: Dict[str, LeafExtractionTask] = {}
    # Ensure we have a leaf task for each base table and join table
    for t in base_tables:
        _ensure_leaf(leaf_map, t["name"], t.get("alias"))
    for j in joins:
        _ensure_leaf(leaf_map, j.table, j.alias)

    #SELECT columns
    for item in select_items:
        for col in item.columns:
            if col.table_name is None:
                continue
            _ensure_leaf(leaf_map, col.table_name, col.table_alias)
            _add_unique_str(leaf_map[col.table_name].columns, col.column_name)
            _add_unique_str(leaf_map[col.table_name].select_columns, col.column_name)
            if item.is_aggregate:
                _add_unique_str(leaf_map[col.table_name].aggregate_columns, col.column_name)

    #GROUP BY columns
    for g in group_by:
        for col in g.columns:
            if col.table_name is None:
                continue
            _ensure_leaf(leaf_map, col.table_name, col.table_alias)
            _add_unique_str(leaf_map[col.table_name].columns, col.column_name)
            _add_unique_str(leaf_map[col.table_name].group_by_columns, col.column_name)

    #JOIN columns
    for j in joins:
        for col in j.on_columns:
            if col.table_name is None:
                continue
            _ensure_leaf(leaf_map, col.table_name, col.table_alias)
            _add_unique_str(leaf_map[col.table_name].columns, col.column_name)
            _add_unique_str(leaf_map[col.table_name].join_keys, col.column_name)

    #Local predicates
    #A predicate is local if it only involves one table and does not contain OR conditions (pushable). 
    #Such predicates can be applied at the leaf level during the scan of that table.
    where = tree.args.get("where")
    flat_preds = _flatten_and_conditions(where.this) if where is not None and where.this is not None else []

    for pred_spec, pred_node in zip(predicates, flat_preds):
        if pred_spec.pushable and len(pred_spec.tables) == 1:
            table_name = pred_spec.tables[0]
            _ensure_leaf(leaf_map, table_name, None)

            normalized_pred_sql = _normalize_predicate_sql(pred_node, alias_map)
            _add_unique_str(leaf_map[table_name].local_predicates, normalized_pred_sql)

            for col in pred_spec.columns:
                if col.table_name == table_name:
                    _add_unique_str(leaf_map[table_name].columns, col.column_name)
        #Leaf or FilterLeaf scan assigned
    leaf_tasks = _finalize_leaf_tasks(leaf_map)

    #Build post-ops
    post_ops: List[PostOp] = []

    if group_by:
        post_ops.append(
            PostOp(
                op="GroupBy",
                payload={"keys": [g.sql for g in group_by]},
            )
        )

    if aggregates:
        post_ops.append(
            PostOp(
                op="Aggregate",
                payload={"aggregates": [asdict(a) for a in aggregates]},
            )
        )

    post_ops.append(
        PostOp(
            op="Project",
            payload={"select": [asdict(s) for s in select_items]},
        )
    )

    if order_by:
        post_ops.append(
            PostOp(
                op="OrderBy",
                payload={
                    "items": [asdict(o) for o in order_by],
                },
            )
        )

    if limit is not None:
        post_ops.append(
            PostOp(
                op="Limit",
                payload={"value": limit},
            )
        )
    #return the final query plan
    return QueryPlan(
        query_type="SELECT",
        sql=sql,
        leaf_tasks=leaf_tasks,
        joins=joins,
        post_ops=post_ops,
    )


# =========================
# Example
# =========================

if __name__ == "__main__":

    sql = """
    SELECT n.n_name, COUNT(*) as num_customers
    FROM nation n  
    JOIN customer c ON n.n_nationkey = c.c_nationkey
    WHERE n.n_regionkey = 1 AND c.c_acctbal > 1000
    GROUP BY n.n_name
    """

    plan = build_query_plan(sql)
    print(json.dumps(plan.to_dict(), indent=2))
    leaf_tasks = plan.leaf_tasks
    print("Leaf tasks:")
    for task in leaf_tasks:
        print(json.dumps(asdict(task), indent=2))
        