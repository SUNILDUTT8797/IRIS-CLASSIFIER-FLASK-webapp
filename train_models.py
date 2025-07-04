import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import os

# Load dataset
df = pd.read_csv("dataset/iris.csv")
X = df.drop("species", axis=1)  # Features
y = df["species"]  # Target

# Encode target labels (species)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train models
logit = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC(probability=True)  # Set probability=True for SVM

logit.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Save models
os.makedirs("models", exist_ok=True)  # Ensure models folder exists
joblib.dump(logit, "models/logit_model.pkl")
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")  # Save label encoder for species names
print("models saved!!")
