import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

# Convert it into a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column as 'species' (this is what you're likely trying to use)
df['species'] = iris.target

# Check column names to verify 'species' is added
print(df.columns)

# Save to CSV
df.to_csv('CO22368_iris.csv', index=False)

print("✅ Iris dataset saved as 'CO22368_iris.csv'")
