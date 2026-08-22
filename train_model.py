import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load dataset
data = pd.read_csv("customer_satisfaction.csv")

# Input columns
X = data[
    [
        "Age",
        "Gender",
        "Purchase_Frequency",
        "Service_Quality",
        "Product_Quality",
        "Support_Quality",
        "Waiting_Time",
        "Value_for_Money",
        "Ease_of_Use",
        "Recommendation",
        "Complaint_Resolution"
    ]
]

# Convert Gender into numbers
X["Gender"] = X["Gender"].map({
    "Male": 0,
    "Female": 1
})

# Target column
y = data["Satisfaction"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully!")