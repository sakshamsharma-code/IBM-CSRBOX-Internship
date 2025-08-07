import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import re

# Load your dataset
data = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

# Preprocess text
def clean_text(text):
    return re.sub(r'[^a-zA-Z]', ' ', text).lower()

data['Review'] = data['Review'].apply(clean_text)

# Vectorization with correct number of features
vectorizer = TfidfVectorizer(max_features=1500)
X = vectorizer.fit_transform(data['Review']).toarray()
y = data['Liked']  # Assuming the target column is named 'Liked'

# Train model
model = SVC(kernel='linear', probability=True)
model.fit(X, y)

# Save both model and vectorizer
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
