"""Export the supplied Mathe pg_dump to gateway CSVs without running SQL."""
import argparse
import csv
import json
from pathlib import Path
import re


def decode_copy(value):
    if value == r"\N":
        return ""
    escapes = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}
    return re.sub(r"\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)",
                  lambda m: chr(int(m[1][1:], 16)) if m[1].startswith("x") else
                  chr(int(m[1], 8)) if m[1][0].isdigit() else escapes.get(m[1], m[1]), value)


def prepare(source, output):
    text = source.read_text(encoding="utf-8")
    tables = {}
    for table, body in re.findall(r"CREATE TABLE mathe\.(\w+) \(\n(.*?)\n\);", text, re.S):
        columns = {}
        for line in body.splitlines():
            name, sql_type = line.strip().rstrip(",").split(" ", 1)
            columns[name] = {"type": sql_type.split(" NOT NULL")[0]}
        columns[f"{table}_rownum"] = {"type": "text"}
        tables[table] = {"columns": columns, "primary_key_candidates": []}
    for table, columns in re.findall(
        r"ALTER TABLE ONLY mathe\.(\w+)\s+ADD CONSTRAINT \w+ PRIMARY KEY \(([^)]+)\);", text
    ):
        tables[table]["primary_key_candidates"] = [{"columns": columns.split(", ")}]
    foreign_keys = []
    for table, columns, target, target_columns in re.findall(
        r"ALTER TABLE ONLY mathe\.(\w+)\s+ADD CONSTRAINT \w+ FOREIGN KEY \(([^)]+)\) REFERENCES mathe\.(\w+)\(([^)]+)\)", text
    ):
        foreign_keys.append({"from_table": table, "from_columns": columns.split(", "),
                             "name_similarity": 1.0, "source": "declared_foreign_key",
                             "to_table": target, "to_columns": target_columns.split(", ")})
    output.mkdir(parents=True, exist_ok=True)
    exported = set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            match = re.fullmatch(r"COPY mathe\.(\w+) \(([^)]+)\) FROM stdin;\n", line)
            if not match:
                continue
            table, columns = match[1], match[2].split(", ")
            with (output / f"{table}.csv").open("w", encoding="utf-8", newline="") as dest:
                writer = csv.writer(dest)
                writer.writerow(columns + [f"{table}_rownum"])
                count = 0
                for row in stream:
                    if row.rstrip("\n") == r"\.":
                        break
                    values = row.rstrip("\n").split("\t")
                    if len(values) != len(columns):
                        raise ValueError(f"Invalid COPY row in {table}")
                    count += 1
                    writer.writerow([decode_copy(v) for v in values] + [f"{table}_{count}"])
                else:
                    raise ValueError(f"Unterminated COPY block: {table}")
            tables[table]["row_count"] = count
            exported.add(table)
            print(f"{table}: {count} rows")
    if exported != set(tables):
        raise ValueError("Missing COPY data for one or more tables")
    (output / "schema_profile_mathe.json").write_text(
        json.dumps({"tables": tables, "foreign_key_candidates": foreign_keys}, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("01_mathe_pgsql.sql"))
    parser.add_argument("--output", type=Path, default=Path("mathe_csv"))
    args = parser.parse_args()
    prepare(args.source, args.output)
