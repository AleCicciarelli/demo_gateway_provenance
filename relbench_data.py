import os
import relbench

DATASET = "rel-arxiv"   # oppure "rel-stack"
OUTPUT_DIR = f"{DATASET}_csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = relbench.load_dataset(DATASET)

# IMPORTANT:
# False = prendi l'intero database, non solo la parte precedente
# al test timestamp di RelBench.
db = dataset.get_db(upto_test_timestamp=False)

for table_name, table in db.table_dict.items():

    df = table.df

    print(
        table_name,
        len(df),
        df.columns.tolist()
    )

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{table_name}.csv"
        ),
        index=False
    )