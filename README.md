# Secure Financial-Health Analytics Pipeline & Predictive Engine

An end-to-end, multi-disciplinary data platform that ingests synthetic banking & health data, implements cryptographic privacy safeguards, runs an in-memory SQL data warehouse, evaluates advanced machine learning classifiers under heavy class imbalance, and performs Bayesian posterior inference.

## Architecture Overview

```text
[ Kaggle API ] ──> [ Python Ingestion ] ──> [ SHA-256 PII Hashing ]
                                                   │
                                                   ▼
[ R Bayesian Stats ] <── [ DuckDB SQL Warehouse ] <─┤
                                                   ▼
                                            [ SMOTE Balancing ]
                                                   │
                                                   ▼
                                         [ XGBoost vs Random Forest ]

     