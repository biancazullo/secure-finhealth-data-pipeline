import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb 


print("Loading the Master Feature Matrix...")
df = pd.read_parquet('master_ml_dataset.parquet')

if 'user_id' in df.columns:
    df = df.drop(columns=['user_id'])

TARGET = 'is_fraud_risk'

text_columns = df.select_dtypes(exclude=['number']).columns
for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df = df.fillna(0)

df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧬 Applying SMOTE to synthesize new fraud data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("⚡ Training the XGBoost Classifier...")


model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,    
    max_depth=5,          
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss' 
)

model.fit(X_train_smote, y_train_smote)


print("🎯 Testing XGBoost against the hidden 20%...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\n======================================")
print(f"XGBOOST MODEL ACCURACY: {accuracy * 100:.2f}%")
print(f"======================================\n")

print("Detailed Performance Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nXGBoost Feature Importance (Gain):")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(5))
