"""
Project 2: Data Classification Using AI
----------------------------------------
Goal: Build a basic classification model using a small dataset (commerce.csv).

Task: Predict the OrderStatus (Pending, Shipped, Delivered, Cancelled, Returned)
of an e-commerce order based on details known at order time (product, price,
quantity, payment method, coupon, referral source, etc.)

Key Requirements covered:
  1. Load and understand a dataset
  2. Split data into training and testing sets
  3. Apply a simple classification algorithm
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# STEP 1: Load and understand the dataset
# ---------------------------------------------------------
df = pd.read_csv("commerce.csv", sep="\t")

print("Dataset shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTarget variable (OrderStatus) distribution:")
print(df["OrderStatus"].value_counts())

# ---------------------------------------------------------
# STEP 2: Prepare / clean the data
# ---------------------------------------------------------
# Fill missing coupon codes with "NoCoupon" (a lot of orders had no coupon)
df["CouponCode"] = df["CouponCode"].fillna("NoCoupon")

# Select features that would realistically be known when the order is placed.
# We drop ID-like columns (OrderID, CustomerID, TrackingNumber, ShippingAddress,
# Date) since they are identifiers/text, not predictive signal for this simple model.
feature_cols = [
    "Product", "Quantity", "UnitPrice", "PaymentMethod",
    "ItemsInCart", "CouponCode", "ReferralSource", "TotalPrice"
]
target_col = "OrderStatus"

X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode categorical (text) columns into numbers so the model can use them
categorical_cols = ["Product", "PaymentMethod", "CouponCode", "ReferralSource"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode the target labels too
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# ---------------------------------------------------------
# STEP 3: Split data into training and testing sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")

# ---------------------------------------------------------
# STEP 4: Apply a simple classification algorithm
# ---------------------------------------------------------
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 5: Evaluate the model
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=target_encoder.classes_, columns=target_encoder.classes_)
print(cm_df)

# ---------------------------------------------------------
# STEP 6: Feature importance (what the model found useful)
# ---------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# ---------------------------------------------------------
# STEP 7: Try a prediction on a new, unseen example
# ---------------------------------------------------------
sample_order = pd.DataFrame([{
    "Product": "Laptop",
    "Quantity": 2,
    "UnitPrice": 500.00,
    "PaymentMethod": "Credit Card",
    "ItemsInCart": 5,
    "CouponCode": "SAVE10",
    "ReferralSource": "Instagram",
    "TotalPrice": 1000.00
}])

for col in categorical_cols:
    sample_order[col] = encoders[col].transform(sample_order[col])

prediction = model.predict(sample_order[feature_cols])
predicted_label = target_encoder.inverse_transform(prediction)
print(f"\nSample new order predicted status: {predicted_label[0]}")