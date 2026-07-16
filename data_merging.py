import pandas as pd

print("Loading entities...")

profiles_df = pd.read_csv('SINIOR_MARKET_SAMPLE_v1/Sample_Users.csv') # Entity 1
txns_df = pd.read_csv('SINIOR_MARKET_SAMPLE_v1/Sample_Transactions.csv')          # Entity 2
visits_df = pd.read_csv('SINIOR_MARKET_SAMPLE_v1/Sample_Health_Records.csv')       # Entity 3

print("Aggregating Transaction Data...")
txns_summary = txns_df.groupby('user_id').agg(
    total_spent=('amount', 'sum'),
    txn_count=('transaction_id', 'count'),
    fraud_flags=('is_fraud', 'sum')
).reset_index()

print("Aggregating Healthcare Data...")
visits_summary = visits_df.groupby('user_id').agg(
    total_hospital_cost=('hospital_cost', 'sum'),
    visit_count=('visit_id', 'count')
).reset_index()

print("Joining Entities together...")
master_df = profiles_df.merge(txns_summary, on='user_id', how='left')
master_df = master_df.merge(visits_summary, on='user_id', how='left')

master_df = master_df.fillna(0)


output_file = 'master_ml_dataset.parquet'
master_df.to_parquet(output_file, index=False)

print(f" Success! Master dataset created with shape: {master_df.shape}")
print(f" Saved as: {output_file}")