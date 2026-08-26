from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


LeafRunner = Callable[[Dict[str, Any], str, Dict[str, List[str]], List[str]], Dict[str, Any]]
BaseQueryBuilder = Callable[[Dict[str, Any], bool], str]
LogEvent = Callable[[Dict[str, Any]], None]


@dataclass
class JoinEdge:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    on_sql: str = ""


@dataclass
class IterativeJoinState:
    selected_rows_by_table: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    bindings: Dict[str, Dict[str, Set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    binding_sources: Dict[str, Dict[str, Dict[str, Set[str]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    )
    completed_tables: Set[str] = field(default_factory=set)


def run_iterative_join_pipeline(
    sql_query: str,
    plan: Dict[str, Any],
    dataset: str,
    pipeline_id: str,
    run_leaf: LeafRunner,
    build_base_retrieval_query: BaseQueryBuilder,
    log_event: Optional[LogEvent] = None,
) -> Dict[str, Any]:
    leaf_tasks = [
        task for task in plan.get("leaf_tasks") or []
        if isinstance(task, dict) and _task_table(task)
    ]
    task_by_table = {_task_table(task): task for task in leaf_tasks}
    join_edges = _join_edges_from_plan(plan)
    state = IterativeJoinState()
    leaf_outputs: List[Dict[str, Any]] = []

    step = 0
    while len(state.completed_tables) < len(task_by_table):
        task = _choose_next_leaf(leaf_tasks, state)
        if task is None:
            break

        table = _task_table(task)
        inherited = _bindings_for_table(state, table)
        source_row_ids = _source_row_ids_for_table(state, table)
        source_row_summaries = _source_row_summaries_for_table(state, table)
        retrieval_query = _build_join_step_retrieval_query(
            task=task,
            inherited_bindings=inherited,
            source_row_ids=source_row_ids,
            source_row_summaries=source_row_summaries,
            base_query=build_base_retrieval_query(task, True),
        )

        step += 1
        if log_event:
            log_event({
                "type": "iterative_join_leaf_start",
                "dataset": dataset,
                "sql_query": sql_query,
                "step": step,
                "table": table,
                "retrieval_query": retrieval_query,
                "bindings": inherited,
                "source_row_ids": source_row_ids,
                "source_row_summaries": source_row_summaries,
            })

        leaf_output = run_leaf(task, retrieval_query, inherited, source_row_ids)
        leaf_output["pipeline"] = pipeline_id
        leaf_output["iterative_join"] = {
            "step": step,
            "inherited_bindings": inherited,
            "source_row_ids": source_row_ids,
            "source_row_summaries": source_row_summaries,
        }

        leaf_outputs.append(leaf_output)
        _record_selected_rows(state, table, leaf_output)
        _propagate_bindings(state, table, join_edges)
        state.completed_tables.add(table)

        if log_event:
            parsed_rows = leaf_output.get("parsed_output") or []
            log_event({
                "type": "iterative_join_leaf_done",
                "dataset": dataset,
                "sql_query": sql_query,
                "step": step,
                "table": table,
                "rows": len(parsed_rows) if isinstance(parsed_rows, list) else 0,
                "bindings": _serializable_bindings(state.bindings),
            })

    return {
        "dataset": dataset,
        "sql": sql_query,
        "plan": plan,
        "leaf_outputs": leaf_outputs,
        "iterative_join": {
            "join_edges": [edge.__dict__ for edge in join_edges],
            "bindings": _serializable_bindings(state.bindings),
            "completed_tables": sorted(state.completed_tables),
        },
    }


def _task_table(task: Dict[str, Any]) -> str:
    return str(task.get("table_name") or task.get("table") or "").strip()


def _join_edges_from_plan(plan: Dict[str, Any]) -> List[JoinEdge]:
    edges: List[JoinEdge] = []
    for join in plan.get("joins") or []:
        if not isinstance(join, dict):
            continue
        columns = [
            col for col in join.get("on_columns") or []
            if isinstance(col, dict) and col.get("table_name") and col.get("column_name")
        ]
        for index in range(0, len(columns) - 1, 2):
            left = columns[index]
            right = columns[index + 1]
            edges.append(
                JoinEdge(
                    left_table=str(left["table_name"]),
                    left_column=str(left["column_name"]),
                    right_table=str(right["table_name"]),
                    right_column=str(right["column_name"]),
                    on_sql=str(join.get("on_sql") or ""),
                )
            )
    return edges


def _choose_next_leaf(
    leaf_tasks: List[Dict[str, Any]],
    state: IterativeJoinState,
) -> Optional[Dict[str, Any]]:
    remaining = [
        task for task in leaf_tasks
        if _task_table(task) and _task_table(task) not in state.completed_tables
    ]
    if not remaining:
        return None

    def score(task: Dict[str, Any]) -> Tuple[int, int, int]:
        table = _task_table(task)
        binding_count = sum(len(values) for values in state.bindings.get(table, {}).values())
        predicate_count = len(task.get("local_predicates") or [])
        join_key_count = len(task.get("join_keys") or [])
        return (1 if binding_count else 0, predicate_count + binding_count, join_key_count)

    return max(remaining, key=score)


def _bindings_for_table(
    state: IterativeJoinState,
    table: str,
) -> Dict[str, List[str]]:
    return {
        column: sorted(values)
        for column, values in state.bindings.get(table, {}).items()
        if values
    }


def _source_row_ids_for_table(state: IterativeJoinState, table: str) -> List[str]:
    row_ids: Set[str] = set()
    for values_by_column in state.binding_sources.get(table, {}).values():
        for source_ids in values_by_column.values():
            row_ids.update(source_ids)
    return sorted(row_ids)


def _source_row_summaries_for_table(
    state: IterativeJoinState,
    table: str,
    max_rows: int = 4,
    max_columns: int = 8,
) -> List[str]:
    source_ids = _source_row_ids_for_table(state, table)
    if not source_ids:
        return []

    summaries: List[str] = []
    for source_id in source_ids:
        found = _find_selected_row_by_id(state, source_id)
        if found is None:
            continue
        source_table, row = found
        pieces = []
        for column, value in row.items():
            if column == "__rid__" or str(column).endswith("_rownum"):
                continue
            text = str(value).strip()
            if not text:
                continue
            pieces.append(f"{source_table}.{column} = {text}")
            if len(pieces) >= max_columns:
                break
        if pieces:
            summaries.append(f"{source_id}: " + "; ".join(pieces))
        if len(summaries) >= max_rows:
            break
    return summaries


def _find_selected_row_by_id(
    state: IterativeJoinState,
    row_id: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    for table, rows in state.selected_rows_by_table.items():
        row = rows.get(row_id)
        if row is not None:
            return table, row
    return None


def _build_join_step_retrieval_query(
    task: Dict[str, Any],
    inherited_bindings: Dict[str, List[str]],
    source_row_ids: List[str],
    source_row_summaries: List[str],
    base_query: str,
) -> str:
    table = _task_table(task)
    parts = [f"Table: {table}" if table else _clean_retrieval_query(base_query)]

    columns = _dedupe_strings(
        [
            *(task.get("select_columns") or []),
            *(task.get("join_keys") or []),
            *(task.get("group_by_columns") or []),
            *(task.get("aggregate_columns") or []),
            *(task.get("columns") or []),
        ]
    )
    if columns:
        parts.append("Columns: " + ", ".join(columns))

    predicates = _dedupe_strings(task.get("local_predicates") or [])
    if predicates:
        parts.append("Filters: " + " and ".join(predicates))

    binding_fragments = []
    for column, values in inherited_bindings.items():
        if len(values) == 1:
            binding_fragments.append(f"{column} = {values[0]}")
        elif values:
            binding_fragments.append(f"{column} in ({', '.join(values)})")

    if binding_fragments:
        parts.append("Join filter: " + " and ".join(binding_fragments))

    # Source row ids and summaries are provenance, not retrieval criteria. They
    # remain in iterative state and UI events but are deliberately excluded
    # from the text embedded by the semantic retriever.
    return ". ".join(part for part in parts if part)


def _clean_retrieval_query(query: str) -> str:
    return (
        str(query or "").strip()
        .replace(". where ", " where ")
        .replace(". needed columns:", "; needed columns:")
    )


def _record_selected_rows(
    state: IterativeJoinState,
    table: str,
    leaf_output: Dict[str, Any],
) -> None:
    for item in leaf_output.get("parsed_output") or []:
        if not isinstance(item, dict):
            continue
        row_id = item.get("row_id")
        values = item.get("values")
        if isinstance(row_id, str) and isinstance(values, dict):
            state.selected_rows_by_table[table][row_id] = values


def _propagate_bindings(
    state: IterativeJoinState,
    table: str,
    join_edges: List[JoinEdge],
) -> None:
    selected_rows = state.selected_rows_by_table.get(table, {})
    for row_id, row in selected_rows.items():
        for edge in join_edges:
            if table == edge.left_table and edge.left_column in row:
                _add_binding(
                    state,
                    target_table=edge.right_table,
                    target_column=edge.right_column,
                    value=row[edge.left_column],
                    source_row_id=row_id,
                )
            if table == edge.right_table and edge.right_column in row:
                _add_binding(
                    state,
                    target_table=edge.left_table,
                    target_column=edge.left_column,
                    value=row[edge.right_column],
                    source_row_id=row_id,
                )


def _add_binding(
    state: IterativeJoinState,
    target_table: str,
    target_column: str,
    value: Any,
    source_row_id: str,
) -> None:
    value_text = str(value).strip()
    if not value_text:
        return
    state.bindings[target_table][target_column].add(value_text)
    state.binding_sources[target_table][target_column][value_text].add(source_row_id)


def _serializable_bindings(
    bindings: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    return {
        table: {
            column: sorted(values)
            for column, values in columns.items()
            if values
        }
        for table, columns in bindings.items()
    }


def _dedupe_strings(values: List[Any]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
