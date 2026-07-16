import duckdb
import pandas as pd

print("Booting up DuckDB Local Data Warehouse...")
con = duckdb.connect(database=':memory:')

print("Executing SQL Query on Parquet File...\n")

sql_query = """
    SELECT 
        is_fraud_risk,
        COUNT(*) as total_users,
        ROUND(AVG(annual_income), 2) as avg_income,
        ROUND(AVG(total_hospital_cost), 2) as avg_hospital_cost,
        SUM(fraud_flags) as total_txn_fraud_flags
    FROM 'master_ml_dataset.parquet'
    GROUP BY is_fraud_risk
    ORDER BY is_fraud_risk DESC;
"""

results_df = con.execute(sql_query).df()

print("======================================================")
print("SQL AGGREGATION RESULTS:")
print("======================================================")
print(results_df.to_string(index=False))
print("======================================================\n")

results_df.to_csv("bayesian_input_data.csv", index=False)
print("Query results exported to 'bayesian_input_data.csv'")