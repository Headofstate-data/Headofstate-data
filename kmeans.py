import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
striker = pd.read_excel("Strikers_performance.xlsx")

# Strip column names to remove unwanted spaces
striker.columns = striker.columns.str.strip()

# Check for missing values
print("Missing Data:\n", striker.isnull().sum())

# Impute missing values with mean
for col in ['Movement off the Ball', 'Penalty Success Rate', 'Big Game Performance']:
    if col in striker.columns:
        imputer = SimpleImputer(strategy="mean")
        striker[col] = imputer.fit_transform(striker[[col]])

# Create 'Total Contribution Score'
if all(col in striker.columns for col in [
    'Goals Scored', 'Assists', 'Shots on Target', 'Dribbling Success',
    'Aerial Duels Won', 'Defensive Contribution', 'Big Game Performance', 'Consistency']):
    striker['Total_Contribution_Score'] = (
            striker['Goals Scored'] + striker['Assists'] + striker['Shots on Target'] +
            striker['Dribbling Success'] + striker['Aerial Duels Won'] +
            striker['Defensive Contribution'] + striker['Big Game Performance'] +
            striker['Consistency']
    )

# K-Means Clustering
df_cluster = striker.drop(columns=['Striker_ID'])
df_cluster = pd.get_dummies(df_cluster, drop_first=True)

# Elbow Method for Optimal Clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(df_cluster)
    wcss.append(kmeans.inertia_)

# Plot elbow chart
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Chart')
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.show()

# Choose optimal clusters (assuming elbow point is 2)
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, n_init='auto', random_state=42)
striker['Clusters'] = kmeans.fit_predict(df_cluster)

# Group by cluster
if 'Total_Contribution_Score' in striker.columns:
    print("Average Total Contribution Score per Cluster:\n",
          striker.groupby('Clusters')['Total_Contribution_Score'].mean())

# Assign cluster labels
striker['Strikers types'] = striker['Clusters'].map({0: 'Best strikers', 1: 'Regular strikers'})

# One-Hot Encoding for categorical variables
striker = pd.get_dummies(striker, columns=['Footedness', 'Marital Status', 'Nationality'], drop_first=True)

# Drop Clusters column
striker.drop(columns=['Clusters'], inplace=True)

# Properly Encode 'Strikers types'
label_encoder = LabelEncoder()
striker['Strikers types'] = label_encoder.fit_transform(striker['Strikers types'])

# Display final dataset
#print(striker.head())

# Logistic Regression ML Model
X = striker.drop(columns=['Striker_ID'])  # Drop Striker_ID only
y = striker['Strikers types']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Model Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print (conf_matrix)
# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Regular strikers', 'Best strikers'],
            yticklabels=['Regular strikers', 'Best strikers'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

