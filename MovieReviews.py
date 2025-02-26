import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

# Dataset path
dataset = "/Users/damiojikutu/Desktop/Data analystics/python/Machine Learning/Movie reviews"

if not os.path.exists(dataset):
    raise FileNotFoundError(f"Dataset folder not found: {dataset}")

# Prepare dataset
texts = []
labels = []
for label, folder in enumerate(['pos', 'neg']):
    folder_path = os.path.join(dataset, folder)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(label)

# Create a DataFrame
data = pd.DataFrame({'text': texts, 'label': labels})

# Define preprocessing function using SpaCy
def preprocess_with_spacy(text):
    doc = nlp(text.lower())  # Convert to lowercase
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_with_spacy)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limit features to 1000
X = vectorizer.fit_transform(data['processed_text']).toarray()
y = data['label']

# Feature reduction with PCA
pca = PCA(n_components=500)  # Reduce dimensions to 500
X_reduced = pca.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

#  Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluate SVM
svm_predictions = svm_model.predict(X_test)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)
print("SVM Confusion Matrix:\n", svm_conf_matrix)
print("\nSVM Classification Report:\n", classification_report(y_test, svm_predictions))



# Evaluate Decision Tree
dt_predictions = dt_model.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
print("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_predictions))

#visualise Decisition Tree
sns. heatmap(dt_conf_matrix , annot=True, cmap="Blues")
plt.title ("Decision Tree aConfusion Matrix:")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#visualise SVM
sns. heatmap(svm_conf_matrix , annot=True, cmap="Blues")
plt.title =("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()