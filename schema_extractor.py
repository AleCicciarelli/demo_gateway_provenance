#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


TPCH_COLUMNS = {
    "region": [
        "r_regionkey", "r_name", "r_comment"
    ],
    "nation": [
        "n_nationkey", "n_name", "n_regionkey", "n_comment"
    ],
    "supplier": [
        "s_suppkey", "s_name", "s_address", "s_nationkey",
        "s_phone", "s_acctbal", "s_comment"
    ],
    "customer": [
        "c_custkey", "c_name", "c_address", "c_nationkey",
        "c_phone", "c_acctbal", "c_mktsegment", "c_comment"
    ],
    "part": [
        "p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
        "p_size", "p_container", "p_retailprice", "p_comment"
    ],
    "partsupp": [
        "ps_partkey", "ps_suppkey", "ps_availqty",
        "ps_supplycost", "ps_comment"
    ],
    "orders": [
        "o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice",
        "o_orderdate", "o_orderpriority", "o_clerk",
        "o_shippriority", "o_comment"
    ],
    "lineitem": [
        "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
        "l_quantity", "l_extendedprice", "l_discount", "l_tax",
        "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
        "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"
    ],
}


EXPECTED_TPCH_PK = {
    "region": [["r_regionkey"]],
    "nation": [["n_nationkey"]],
    "supplier": [["s_suppkey"]],
    "customer": [["c_custkey"]],
    "part": [["p_partkey"]],
    "partsupp": [["ps_partkey", "ps_suppkey"]],
    "orders": [["o_orderkey"]],
    "lineitem": [["l_orderkey", "l_linenumber"]],
}


EXPECTED_TPCH_FK = [
    ("nation", ["n_regionkey"], "region", ["r_regionkey"]),
    ("supplier", ["s_nationkey"], "nation", ["n_nationkey"]),
    ("customer", ["c_nationkey"], "nation", ["n_nationkey"]),
    ("orders", ["o_custkey"], "customer", ["c_custkey"]),
    ("partsupp", ["ps_partkey"], "part", ["p_partkey"]),
    ("partsupp", ["ps_suppkey"], "supplier", ["s_suppkey"]),
    ("lineitem", ["l_orderkey"], "orders", ["o_orderkey"]),
    ("lineitem", ["l_partkey", "l_suppkey"], "partsupp", ["ps_partkey", "ps_suppkey"]),
]

RELF1_COLUMNS = {
    "races": [
        "raceId", "year", "round", "circuitId", "name", "date", "time"
    ],
    "circuits": [
        "circuitId", "circuitRef", "name", "location", "country", "lat", "lng", "alt"
    ],
    "drivers": [
        "driverId", "driverRef", "code", "forename", "surname", "dob", "nationality"
    ],
    "constructors": [
        "constructorId", "constructorRef", "name", "nationality"
    ],
    "constructor_results": [
        "constructorResultsId", "raceId", "constructorId", "points", "date"
    ],
    "constructor_standings": [
        "constructorStandingsId", "raceId", "constructorId", "points", "position", "wins", "date"
    ],
    "standings": [
        "driverStandingsId", "raceId", "driverId", "points", "position", "wins", "date"
    ],
    "results": [
        "resultId", "raceId", "driverId", "constructorId", "statusId", "number", "grid", "position", "positionOrder", "points", "laps", "milliseconds", "fastestLap", "rank", "date"
    ],
    "qualifying": [
        "qualifyId", "raceId", "driverId", "constructorId", "number", "position", "date"
    ]
}

EXPECTED_RELF1_PK = {
    "races": [["raceId"]],
    "circuits": [["circuitId"]],
    "drivers": [["driverId"]],
    "constructors": [["constructorId"]],
    "constructor_results": [["constructorResultsId"]],
    "constructor_standings": [["constructorStandingsId"]],
    "standings": [["driverStandingsId"]],
    "results": [["resultId"]],
    "qualifying": [["qualifyId"]]
}
EXPECTED_RELF1_FK = [
    ("constructor_results", ["raceId"], "races", ["raceId"]),
    ("constructor_results", ["constructorId"], "constructors", ["constructorId"]),
    ("constructor_standings", ["raceId"], "races", ["raceId"]),
    ("constructor_standings", ["constructorId"], "constructors", ["constructorId"]),
    ("standings", ["raceId"], "races", ["raceId"]),
    ("standings", ["driverId"], "drivers", ["driverId"]),
    ("results", ["raceId"], "races", ["raceId"]),
    ("results", ["driverId"], "drivers", ["driverId"]),
    ("results", ["constructorId"], "constructors", ["constructorId"]),
    ("qualifying", ["raceId"], "races", ["raceId"]),
    ("qualifying", ["driverId"], "drivers", ["driverId"]),
    ("qualifying", ["constructorId"], "constructors", ["constructorId"]),
    ("races", ["circuitId"], "circuits", ["circuitId"])
]
def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def table_name_from_file(path: Path) -> str:
    name = path.stem.lower()
    for table in RELF1_COLUMNS:
        if name == table or name.startswith(table + "_"):
            return table
    return name


def read_csv_table(path: Path, sep: str, relf1_no_header: bool) -> tuple[str, pd.DataFrame]:
    table = table_name_from_file(path)

    if relf1_no_header and table in RELF1_COLUMNS:
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            names=RELF1_COLUMNS[table],
            engine="python",
        )
    else:
        df = pd.read_csv(path, sep=sep, engine="python")

    # Drop possible unnamed trailing column from pipe-separated .tbl files
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df.columns = [str(c).strip() for c in df.columns]
    return table, df


def infer_column_type(series: pd.Series) -> str:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return "unknown"

    numeric = pd.to_numeric(s, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio > 0.95:
        if (numeric.dropna() % 1 == 0).all():
            return "integer"
        return "decimal"

    dates = pd.to_datetime(s, errors="coerce")
    date_ratio = dates.notna().mean()

    if date_ratio > 0.90:
        return "date"

    unique_ratio = s.nunique() / max(len(s), 1)

    if unique_ratio < 0.20:
        return "categorical"

    return "text"

def is_provenance_column(table: str, col: str) -> bool:
    return re.match(r"^.*_rownum$", col.lower().strip()) is not None


def profile_columns(df: pd.DataFrame) -> dict[str, Any]:
    profile = {}

    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        unique = int(s.nunique(dropna=True))
        examples = (
            s.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        profile[col] = {
            "type": infer_column_type(s),
            "is_provenance": is_provenance_column("", col),
            "non_null": non_null,
            "unique": unique,
            "unique_ratio": round(unique / max(len(df), 1), 4),
            "null_ratio": round(1 - non_null / max(len(df), 1), 4),
            "examples": examples,
        }

    return profile


def key_like_score(col: str) -> float:
    n = normalize_name(col)

    score = 0.0

    if n.endswith("key") or n.endswith("id"):
        score += 1.0

    if "key" in n or "id" in n or "Id" in n:
        score += 0.5

    return score


def infer_primary_keys(df: pd.DataFrame, max_composite_size: int = 2) -> list[dict[str, Any]]:
    candidates = []

    # Single-column keys
    for col in df.columns:
        if is_provenance_column("", col):
            continue
        s = df[col]
        non_null = s.notna().all()
        unique = s.nunique(dropna=True) == len(df)

        if non_null and unique:
            candidates.append({
                "columns": [col],
                "kind": "single",
                "score": round(1.0 + key_like_score(col), 3),
                "reason": "unique and non-null",
            })

    # Composite keys, only among key-like columns
    key_like_cols = [
        c for c in df.columns
        if is_composite_key_candidate_column(df, c)
    ]

    for size in range(2, max_composite_size + 1):
        for cols in itertools.combinations(key_like_cols, size):
            unique = df[list(cols)].drop_duplicates().shape[0] == len(df)
            non_null = df[list(cols)].notna().all().all()

            if unique and non_null:
                candidates.append({
                    "columns": list(cols),
                    "kind": "composite",
                    "score": round(0.9 + sum(key_like_score(c) for c in cols) / size, 3),
                    "reason": "composite unique and non-null",
                })

    candidates = sorted(candidates, key=lambda x: (-x["score"], len(x["columns"])))
    return candidates[:10]


def value_set(series: pd.Series, sample_limit: int | None = None) -> set[str]:
    s = series.dropna().astype(str).str.strip()

    if sample_limit is not None and len(s) > sample_limit:
        s = s.sample(sample_limit, random_state=42)

    return set(s.tolist())


def name_similarity(a: str, b: str) -> float:
    na = normalize_name(a)
    nb = normalize_name(b)

    if na == nb:
        return 1.0

    # c_nationkey and n_nationkey share nationkey
    suffix_a = re.sub(r"^[a-z]+", "", na)
    suffix_b = re.sub(r"^[a-z]+", "", nb)

    if suffix_a and suffix_a == suffix_b:
        return 0.8

    if na in nb or nb in na:
        return 0.6

    common = set(re.findall(r"[a-z]+", a.lower())) & set(re.findall(r"[a-z]+", b.lower()))
    if common:
        return 0.4

    return 0.0


def infer_single_column_foreign_keys(
    tables: dict[str, pd.DataFrame],
    pk_candidates: dict[str, list[dict[str, Any]]],
    min_coverage: float = 0.95,
    min_name_similarity: float = 0.0,
    sample_limit: int | None = 20000,
) -> list[dict[str, Any]]:
    fks = []

    target_pks = []

    for target_table, candidates in pk_candidates.items():
        for cand in candidates:
            if len(cand["columns"]) == 1:
                if is_provenance_column(target_table, cand["columns"][0]):
                    continue
                target_pks.append((target_table, cand["columns"][0]))

    for source_table, source_df in tables.items():
        for source_col in source_df.columns:
            if is_provenance_column(source_table, source_col):
                continue
            source_values = value_set(source_df[source_col], sample_limit=sample_limit)

            if not source_values:
                continue

            for target_table, target_col in target_pks:
                if source_table == target_table:
                    continue

                target_values = value_set(tables[target_table][target_col], sample_limit=None)

                overlap = len(source_values & target_values)
                coverage = overlap / max(len(source_values), 1)

                if coverage >= min_coverage:
                    sim = name_similarity(source_col, target_col)
                    if sim <= min_name_similarity:
                        continue

                    score = 0.75 * coverage + 0.25 * sim

                    fks.append({
                        "from_table": source_table,
                        "from_columns": [source_col],
                        "to_table": target_table,
                        "to_columns": [target_col],
                        "coverage": round(coverage, 4),
                        "name_similarity": round(sim, 4),
                        "score": round(score, 4),
                    })

    fks = sorted(fks, key=lambda x: -x["score"])
    return fks
def is_composite_key_candidate_column(df: pd.DataFrame, col: str) -> bool:
    if is_provenance_column("", col):
        return False

    s = df[col]
    non_null_ratio = s.notna().mean()

    if non_null_ratio < 0.95:
        return False

    col_type = infer_column_type(s)

    # Examples: comments, reviews, addresses.
    if col_type == "text":
        avg_len = s.dropna().astype(str).str.len().mean()
        if avg_len > 30:
            return False

    unique = s.nunique(dropna=True)

    # Constant columns are useless in a key.
    if unique <= 1:
        return False

    return True

def infer_composite_foreign_keys(
    tables: dict[str, pd.DataFrame],
    pk_candidates: dict[str, list[dict[str, Any]]],
    min_coverage: float = 0.95,
    min_name_similarity: float = 0.0,
    max_source_pairs: int = 20,
) -> list[dict[str, Any]]:
    fks = []

    target_composite_pks = []

    for target_table, candidates in pk_candidates.items():
        for cand in candidates:
            if len(cand["columns"]) == 2:
                target_composite_pks.append((target_table, cand["columns"]))

    for source_table, source_df in tables.items():
        source_key_cols = [c for c in source_df.columns if key_like_score(c) > 0 and not is_provenance_column(source_table, c)]
        source_pairs = list(itertools.combinations(source_key_cols, 2))[:max_source_pairs]

        for source_cols in source_pairs:
            source_pairs_values = set(
                map(tuple, source_df[list(source_cols)].dropna().astype(str).values.tolist())
            )

            if not source_pairs_values:
                continue

            for target_table, target_cols in target_composite_pks:
                if source_table == target_table:
                    continue

                target_pairs_values = set(
                    map(tuple, tables[target_table][target_cols].dropna().astype(str).values.tolist())
                )

                overlap = len(source_pairs_values & target_pairs_values)
                coverage = overlap / max(len(source_pairs_values), 1)

                if coverage >= min_coverage:
                    sim = sum(
                        name_similarity(a, b)
                        for a, b in zip(source_cols, target_cols)
                    ) / len(source_cols)

                    if sim <= min_name_similarity:
                        continue

                    score = 0.75 * coverage + 0.25 * sim

                    fks.append({
                        "from_table": source_table,
                        "from_columns": list(source_cols),
                        "to_table": target_table,
                        "to_columns": list(target_cols),
                        "coverage": round(coverage, 4),
                        "name_similarity": round(sim, 4),
                        "score": round(score, 4),
                    })

    fks = sorted(fks, key=lambda x: -x["score"])
    return fks


def compare_with_relf1(profile: dict[str, Any]) -> dict[str, Any]:
    inferred_pk = {
        table: [frozenset(cand["columns"]) for cand in data["primary_key_candidates"]]
        for table, data in profile["tables"].items()
    }

    inferred_fk = set()
    for fk in profile["foreign_key_candidates"]:
        inferred_fk.add((
            fk["from_table"],
            tuple(fk["from_columns"]),
            fk["to_table"],
            tuple(fk["to_columns"]),
        ))

    expected_pk = {
        table: [frozenset(cols) for cols in keys]
        for table, keys in EXPECTED_RELF1_PK.items()
    }

    expected_fk = set(
        (src_t, tuple(src_c), tgt_t, tuple(tgt_c))
        for src_t, src_c, tgt_t, tgt_c in EXPECTED_RELF1_FK
    )

    pk_report = {}

    for table, expected_keys in expected_pk.items():
        found = inferred_pk.get(table, [])
        pk_report[table] = {
            "expected": [list(x) for x in expected_keys],
            "found_candidates": [list(x) for x in found],
            "matched": any(x in found for x in expected_keys),
        }

    fk_report = {
        "expected": [
            {
                "from": f"{src_t}.{'.'.join(src_c)}",
                "to": f"{tgt_t}.{'.'.join(tgt_c)}",
                "matched": (src_t, tuple(src_c), tgt_t, tuple(tgt_c)) in inferred_fk,
            }
            for src_t, src_c, tgt_t, tgt_c in EXPECTED_RELF1_FK
        ],
        "extra_inferred": [
            {
                "from": f"{src_t}.{'.'.join(src_c)}",
                "to": f"{tgt_t}.{'.'.join(tgt_c)}",
            }
            for src_t, src_c, tgt_t, tgt_c in sorted(inferred_fk - expected_fk)
        ],
    }

    return {
        "primary_keys": pk_report,
        "foreign_keys": fk_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", required=True)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--relf1-no-header", action="store_true")
    parser.add_argument("--out", default="schema_profile.json")
    parser.add_argument("--compare-relf1", action="store_true")
    parser.add_argument("--min-fk-coverage", type=float, default=0.95)
    parser.add_argument("--min-fk-name-similarity", type=float, default=0.0)
    args = parser.parse_args()
    '''how to run:
    python3 schema_extractor.py --csv_dir ./rel-f1-csv/ --sep "," --out ./rel-f1-csv/schema_profile_relf1.json --compare-relf1
    '''
    csv_dir = Path(args.csv_dir)

    paths = sorted(
        list(csv_dir.glob("*.csv")) +
        list(csv_dir.glob("*.tbl"))
    )

    if not paths:
        raise FileNotFoundError(f"No CSV/TBL files found in {csv_dir}")

    tables: dict[str, pd.DataFrame] = {}

    for path in paths:
        table, df = read_csv_table(path, sep=args.sep, relf1_no_header=args.relf1_no_header)
        tables[table] = df
        print(f"Loaded {table}: {df.shape[0]} rows, {df.shape[1]} columns")

    table_profiles = {}
    pk_candidates = {}

    for table, df in tables.items():
        columns_profile = profile_columns(df)
        pks = infer_primary_keys(df)

        pk_candidates[table] = pks

        table_profiles[table] = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "columns": columns_profile,
            "primary_key_candidates": pks,
        }

    single_fks = infer_single_column_foreign_keys(
        tables,
        pk_candidates,
        min_coverage=args.min_fk_coverage,
        min_name_similarity=args.min_fk_name_similarity,
    )

    composite_fks = infer_composite_foreign_keys(
        tables,
        pk_candidates,
        min_coverage=args.min_fk_coverage,
        min_name_similarity=args.min_fk_name_similarity,
    )

    profile = {
        "tables": table_profiles,
        "foreign_key_candidates": single_fks + composite_fks,
    }

    if args.compare_relf1:
        profile["relf1_comparison"] = compare_with_relf1(profile)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"\nSaved schema profile to: {out_path}")

    if args.compare_relf1:
        print("\nRELF1 comparison summary")

        pk_report = profile["relf1_comparison"]["primary_keys"]
        for table, item in pk_report.items():
            status = "OK" if item["matched"] else "MISSING"
            print(f"PK {table}: {status}")

        print("\nForeign keys:")
        for item in profile["relf1_comparison"]["foreign_keys"]["expected"]:
            status = "OK" if item["matched"] else "MISSING"
            print(f"FK {item['from']} -> {item['to']}: {status}")


if __name__ == "__main__":
    main()
