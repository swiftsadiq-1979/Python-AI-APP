import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Load CSV with correct header and skip meta rows
df = pd.read_csv('exported_data.csv', skiprows=[1, 2])

# 2. Columns to use (omit Name)
features = ["Age", "Gender", "StudyHours", "Attendance", "HomeworkRate", "Participation", "FavoriteSubject"]
X = df[features].copy()
y = df["FinalGrade"]

# 3. Encode categorical features
encoders = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 4. Encode target/class label
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# 5. Train/test split (optional but standard practice)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 7. Save model & encoders for app use
joblib.dump(rf, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_le, "target_le.pkl")

print("Model and encoders successfully saved. You can now use them in Streamlit!")
