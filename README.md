# Data Classification Using AI 🛒📊

A basic **classification model** built on e-commerce order data to predict an order's final **status** — `Pending`, `Shipped`, `Delivered`, `Cancelled`, or `Returned` — based on order details (product, price, payment method, coupon, referral source, etc.).

## 📌 Goal
Build a basic classification model using a small dataset — understanding and applying the fundamentals of supervised learning.

## 🎯 Key Requirements Covered
- ✅ Load and explore the dataset (shape, missing values, class distribution)
- ✅ Split the data into training and testing sets
- ✅ Apply a simple classification algorithm (Decision Tree)
- ✅ Evaluate the model (accuracy, classification report, confusion matrix)

## 📁 Dataset
`commerce.csv` — 1,200 e-commerce orders, with the following columns:

| Column | Description |
|---|---|
| OrderID, CustomerID, TrackingNumber | Unique identifiers (not used in the model) |
| Date, ShippingAddress | Order metadata |
| Product | Item ordered (Laptop, Phone, Tablet, etc.) |
| Quantity, UnitPrice, TotalPrice | Order value details |
| PaymentMethod | Credit Card, Debit Card, Cash, etc. |
| ItemsInCart | Cart size at checkout |
| CouponCode | Discount code applied (if any) |
| ReferralSource | Traffic source (Instagram, Email, Google, etc.) |
| **OrderStatus** | 🎯 Target variable — the value to predict |

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- scikit-learn (`DecisionTreeClassifier`, `train_test_split`, `LabelEncoder`)

## ⚙️ How It Works
1. **Load & Explore** — read the dataset and print its shape, dtypes, missing values, and target class distribution.
2. **Preprocessing** — fill missing `CouponCode` values with `"NoCoupon"`, and convert text columns to numeric using `LabelEncoder`.
3. **Train-Test Split** — 80% of the data for training, 20% for testing (stratified split, to keep each class's ratio consistent).
4. **Model Training** — train a `DecisionTreeClassifier(max_depth=6)`.
5. **Evaluation** — compute accuracy, precision/recall/F1 (classification report), and a confusion matrix.
6. **Feature Importance** — check which features the model found most useful.
7. **Prediction Demo** — build a new sample order and show its predicted status.

## 📊 Results
- **Accuracy:** ~22–23%
- With 5 classes, random guessing would give ~20% accuracy, so the model performs only slightly better than that.
- The most important features turned out to be `UnitPrice` and `TotalPrice`.
- **Insight:** The fields in this dataset (price, product, payment method, etc.) don't strongly correlate with an order's final status — a valid and realistic finding when the data is mostly random/synthetic.

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn
python classification_project.py
```

## 🚀 Possible Improvements
- Try different algorithms (Random Forest, Logistic Regression, KNN) and compare results
- Hyperparameter tuning
- Visualize feature importance as a bar chart
- Engineer more domain-relevant features (e.g., order timing, customer history)

## 📂 Files
```
├── commerce.csv              # Dataset
├── main.py # Main script (load → split → train → evaluate)
└── README.md                 # Project documentation
```

---
*Project 2 — Data Classification Using AI (Data handling, supervised learning basics, model training)*
