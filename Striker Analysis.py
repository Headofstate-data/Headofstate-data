import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


striker = pd.read_excel("Strikers_performance.xlsx")
print(striker.head())

missing_data = striker.isnull().sum()
print(missing_data,striker.dtypes)

Imputer = SimpleImputer(strategy ="mean")
Imputer.fit(striker[['Movement off the Ball']])
striker[['Movement off the Ball']] = Imputer.transform(striker[['Movement off the Ball']])

PenSuc = SimpleImputer(strategy ="mean")
PenSuc.fit(striker[['Penalty Success Rate']])
striker[['Penalty Success Rate']] = PenSuc.transform(striker[['Penalty Success Rate']])

BGP = SimpleImputer(strategy ="mean")
BGP.fit(striker[['Big Game Performance']])
striker[['Big Game Performance']] = BGP.transform(striker[['Big Game Performance']])

# check if any missing value in columns
missing_data = striker.isnull().sum()
print(missing_data,striker.dtypes)

 #descriptive stats and round up to two
descriptive_stats = (round (striker.describe()),2)
print(descriptive_stats)

#percentage analysis
# step 1: calculate Frequency analysis
frequency = striker['Footedness'].value_counts()

# step 2: calculate percentage  using Frequency
percentage = frequency/len(striker['Footedness'])*100

# print value of frequency and percentage
print ("frequency",  frequency )
print ("percentage ",  percentage )


# draw pie chart
plt.figure(figsize=(6, 6))
percentage.plot(kind = 'pie', autopct = '% 1.2f%%')
plt.title('Footedness')
plt.ylabel("")
plt.show()

# countplot of players' footedness across different nationalities
sns.countplot(x='Footedness',hue = 'Nationality', data = striker)
plt.show()

# nationality strikers have the highest average number of goals scored

avg_goals_nation = striker.groupby ('Nationality') ['Goals Scored']. mean()
print( "Average goals per nation" ,avg_goals_nation )

avg_conv_rate = striker.groupby ('Footedness') ['Conversion Rate'].mean()
print("Average Conversion rate", avg_conv_rate)

#shapiro test
shap_result ={}
for Nation in striker['Nationality'].unique():
    stat, p_value = shapiro(striker.loc[striker["Nationality"] == Nation, "Consistency"])
    shap_result[Nation]= round(p_value,3)
    print ("Shapiro Test :", shap_result)

#levene test
groups = [striker.loc[striker["Nationality"] == nation, "Consistency"] for nation in striker["Nationality"].unique()]
stat, p = levene(*groups)
print(f"Levene’s test p-value: {p}")

#one way anova
t_stats, p_value = f_oneway (*groups)
print('One way anova p-value:', p_value )

# using pearsons correlation analysis to check for corelation between hold up play and consistency

corr, p_value = pearsonr(striker['Consistency'], striker['Hold-up Play'])
print ('pearsons p_value', p_value)

# Using Linear Regression Analysis to check if strikers' hold-up play significantly influences their consistency rate.

x = striker ['Hold-up Play']
y= striker ['Consistency']

x_and_constant = sm.add_constant(x)

model = sm.OLS(y,x_and_constant).fit()

print(model.summary())

#feature encoding

striker['New Feature'] = (striker['Goals Scored'] + striker['Assists'] + striker['Shots on Target'] + striker['Dribbling Success'] + striker['Aerial Duels Won'] + striker['Defensive Contribution'] + striker['Big Game Performance'] + striker['Consistency'] )
encoder = LabelEncoder()
striker ['encoded_Feature'] = encoder.fit_transform (striker['Footedness'])
striker ['encoded_feature_01'] = encoder.fit_transform (striker['Marital Status'])

# creating dummies for Nationality and adding them to the data
dummies = pd.get_dummies(striker['Nationality'])
striker = pd.concat([striker,dummies],axis=1)

print(striker.head())

#k means clusters

# Drop Striker_ID
df_cluster = striker.drop(columns=['Striker_ID'])

# Convert categorical variables into numerical (if any)
df_cluster = pd.get_dummies(df_cluster, drop_first=True)

# Calculate WCSS (Within-Cluster-Sum-of-Squares)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)  # Fixed n_init issue
    kmeans.fit(df_cluster)
    wcss_values = kmeans.inertia_  # Corrected from Inertia_ to inertia_
    wcss.append(wcss_values)

# Plot elbow chart
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Chart')
plt.xlabel('Number of clusters')
plt.ylabel("WCSS")
plt.show()

#Choose optimal clusters (Assuming elbow point is at 2)
optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
striker['Clusters'] = kmeans.fit_predict(df_cluster)

# Calculate average total contribution score per cluster
#avg_contributions = striker.groupby('Clusters')['Total_Contribution_Score'].mean()

# Assign cluster labels
striker['Strikers types'] = striker['Clusters'].map({0: 'Best strikers', 1: 'Regular strikers'})

# Drop the Clusters variable
striker.drop(columns=['Clusters'], inplace=True)

# Feature mapping
striker['Strikers types'] = striker['Strikers types'].map({'Best strikers': 1, 'Regular strikers': 0})

# Display final dataset
print(striker.head())

#Logistic Regression Model
X = striker.drop (striker['Striker_ID'])
y = striker ['Strikers types']

#feature scaling with Standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train 80% and test 20%

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size= 0.2, random_state= 42)

#Train Logistic Regression models
lg_reg = LogisticRegression()
lg_reg.fit(X_train, y_train)

#make predictions
y_pred = lg_reg.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

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

