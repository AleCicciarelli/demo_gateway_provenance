from typing import Dict, Any

# schema info for the TPCH tables, to be used in the prompt. 
SCHEMA_INFO: Dict[str, Any] = {
    "region": {
        "columns": ["r_regionkey", "r_name", "r_comment"],
        "primary_key": "r_regionkey",
        "row_id": "region_rownum",
        "foreign_keys": {}
    },
    "nation": {
        "columns": ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        "primary_key": "n_nationkey",
        "row_id": "nation_rownum",
        "foreign_keys": {
            "n_regionkey": "region.r_regionkey"
        }
    },
    "customer": {
        "columns": [
            "c_custkey", "c_name", "c_address", "c_nationkey",
            "c_phone", "c_acctbal", "c_mktsegment", "c_comment"
        ],
        "primary_key": "c_custkey",
        "row_id": "customer_rownum",
        "foreign_keys": {
            "c_nationkey": "nation.n_nationkey"
        }
    },
    "orders": {
        "columns": [
            "o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice",
            "o_orderdate", "o_orderpriority", "o_clerk",
            "o_shippriority", "o_comment"
        ],
        "primary_key": "o_orderkey",
        "row_id": "orders_rownum",
        "foreign_keys": {
            "o_custkey": "customer.c_custkey"
        }
    },
    "lineitem": {
        "columns": [
            "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber",
            "l_quantity", "l_extendedprice", "l_discount", "l_tax",
            "l_returnflag", "l_linestatus", "l_shipdate",
            "l_commitdate", "l_receiptdate", "l_shipinstruct",
            "l_shipmode", "l_comment"
        ],
        "primary_key": ["l_orderkey", "l_linenumber"],
        "row_id": "lineitem_rownum",
        "foreign_keys": {
            "l_orderkey": "orders.o_orderkey",
            "l_partkey": "part.p_partkey",
            "l_suppkey": "supplier.s_suppkey"
        }
    },
    "supplier": {
        "columns": [
            "s_suppkey", "s_name", "s_address", "s_nationkey",
            "s_phone", "s_acctbal", "s_comment"
        ],
        "primary_key": "s_suppkey",
        "row_id": "supplier_rownum",
        "foreign_keys": {
            "s_nationkey": "nation.n_nationkey"
        }
    },
    "part": {
        "columns": [
            "p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
            "p_size", "p_container", "p_retailprice", "p_comment"
        ],
        "primary_key": "p_partkey",
        "row_id": "part_rownum",
        "foreign_keys": {}
    },
    "partsupp": {
        "columns": [
            "ps_partkey", "ps_suppkey", "ps_availqty",
            "ps_supplycost", "ps_comment"
        ],
        "primary_key": ["ps_partkey", "ps_suppkey"],
        "row_id": "partsupp_rownum",
        "foreign_keys": {
            "ps_partkey": "part.p_partkey",
            "ps_suppkey": "supplier.s_suppkey"
        }
    }
}