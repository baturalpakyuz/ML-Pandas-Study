import time
import requests
import psycopg
import sys
import os

sys.path.append(os.path.abspath(".."))

import config
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Design
# hostid pull
def unique_hostid():
    payload = {
        "jsonrpc": "2.0",
        "method": "host.get",
        "params": {
            "output": ["hostid"]
        },
        "id": 1
    }

    api_headers = {
        "Content-Type": "application/json-rpc",
        "Authorization": f"Bearer {config.PERMANENT_API_TOKEN}"
    }
    # Main Request
    response = requests.post(config.API_CONFIG["url"], json=payload, headers=api_headers, timeout=10) 
    # Convert response to dict
    data = response.json()
    # hostid list
    hostids = [host["hostid"] for host in data["result"]]
    return hostids


# (datapul) Postgresql connection using config config
    #join function
        #output of it to be written using Apache Parquet
def execute_db_call(value, partition_columns=["tag"]):
    """Query DB and store result into Apache Parquet, partitioned dynamically."""

    print("\n📦 Connecting to PostgreSQL Database...")

    query = """
    SELECT
        i.name,
        his.value,
        his.clock,
        it.tag
    FROM history his
    JOIN items i
        ON his.itemid = i.itemid
    JOIN hosts h
        ON i.hostid = h.hostid
    JOIN item_tag it
        ON it.itemid = i.itemid
    WHERE i.type IN (5, 3, 0)
      AND h.hostid = %s
      AND his.clock BETWEEN 1779534243 AND 1779817579
    """

    try:
        with psycopg.connect(**config.DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (value,))
                rows = cur.fetchall()

        print(f"✅ DB Success! Rows fetched: {len(rows)}")
        
        # Guard clause: If no rows returned, exit early
        if not rows:
            print("⚠️ No data found for this host. Skipping parquet write.")
            return None

        # -----------------------------
        # Convert to DataFrame
        # -----------------------------
        df = pd.DataFrame(rows, columns=["name", "value", "clock", "tag"])
        
        # Add hostid so it is available as a column (critical if you want to partition by it!)
        df["hostid"] = value 

        # -----------------------------
        # Write to Parquet (Parametric Partitioning)
        # -----------------------------
        output_directory = "output_data" 
        table = pa.Table.from_pandas(df)

        # We pass the dynamic parameter here
        pq.write_to_dataset(
            table,
            root_path=output_directory,
            partition_cols=partition_columns, 
            compression="snappy"
        )

        print(f"💾 Saved partitioned Parquet to: {output_directory} using partitions: {partition_columns}")

        return df

    except Exception as e:
        print(f"❌ DB Connection failure: {e}")

if __name__ == "__main__":
    hostids_list = unique_hostid()
    for host_id in hostids_list:
        execute_db_call(host_id, partition_columns=["hostid", "tag"])
    


# 
# history his, items i, hosts h, item_tag it


# select his.itemid, his.value, his.clock, it.tag
# where his.clock <> (manuel), i.type == 5,3,0, h.hostid = {value}
# join h.hostid on i.hostid, it.itemid on i.itemid, h.itemid on i.itemid


# main
    # Do you want to data to be pulled
    # if statement
        # yes: 
            # (datapul)
            # class ml
        # no:
            # class ml

# Database system is like this: hosts table have uuid's and name. item_tag table have itemid's and tag values. items table has itemid and hostid values. 