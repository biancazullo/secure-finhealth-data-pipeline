import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 


df = pd.read_parquet('master_ml_dataset.parquet')

if 'user_id' in df.columns:
    df = df.drop(columns=['user_id'])

TARGET = 'is_fraud_risk'

text_columns = df.select_dtypes(exclude=['number']).columns
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df = df.fillna(0)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Original training data: {len(X_train)} rows.")

print("Applying SMOTE to synthesize new fraud data...")
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Balanced training data: {len(X_train_smote)} rows.")

print("Training the Random Forest on the balanced data...")

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_smote, y_train_smote)

print("Testing the model against the hidden 20%...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\n======================================")
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"======================================\n")

print("Detailed Performance Report:")
print(classification_report(y_test, predictions, zero_division=0))

import csv

print("\nExporting dynamic baseline metrics to CSV...")

lr_acc = locals().get('lr_accuracy', 0.0) 
lr_prec = locals().get('lr_precision', 0.0)
lr_rec = locals().get('lr_recall', 0.0)

rf_acc = locals().get('rf_accuracy', 0.0)
rf_prec = locals().get('rf_precision', 0.0)
rf_rec = locals().get('rf_recall', 0.0)

with open('ml_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model', 'accuracy', 'precision_class_1', 'recall_class_1'])
    if rf_acc > 0:
        writer.writerow(['Random Forest (Baseline)', round(rf_acc, 2), round(rf_prec, 2), round(rf_rec, 2)])

print("✅ 'ml_evaluation_results.csv' successfully created with baseline metrics!")