"""Export a complete RelBench database and its declared relational schema."""
import argparse
import json
from pathlib import Path

import relbench


def export_dataset(name, output_dir):
    dataset = relbench.load_dataset(name)
    db = dataset.get_db(upto_test_timestamp=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = {'tables': {}, 'foreign_key_candidates': [], 'source': name}
    for table_name, table in db.table_dict.items():
        df = table.df
        pk = table.pkey_col
        if pk and (df[pk].isna().any() or not df[pk].is_unique):
            raise ValueError(f'Invalid primary key: {table_name}.{pk}')
        profile['tables'][table_name.lower()] = {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'primary_key_candidates': [{'columns': [pk]}] if pk else [],
        }
        for column, target in table.fkey_col_to_pkey_table.items():
            target_pk = db.table_dict[target].pkey_col
            if target_pk is None:
                raise ValueError(f'Missing referenced primary key for {target}')
            profile['foreign_key_candidates'].append({
                'from_table': table_name.lower(), 'from_columns': [column],
                'to_table': target.lower(), 'to_columns': [target_pk],
                'name_similarity': 1.0,
            })
        path = output_dir / f'{table_name}.csv'
        temporary = path.with_suffix('.csv.tmp')
        df.to_csv(temporary, index=False)
        temporary.replace(path)
        print(f'Exported {table_name}: {len(df):,} rows', flush=True)
    schema_path = output_dir / f"schema_profile_{name.replace('-', '')}.json"
    schema_path.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8')
    print(f'Saved schema to {schema_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='rel-arxiv')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    export_dataset(args.dataset, args.output_dir or Path(f'{args.dataset}_csv'))
