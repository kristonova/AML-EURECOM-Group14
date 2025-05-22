import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

# =============================================
# 1) READ DATA
# =============================================
# Change the path according to the location of your file
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# ---------------------------------------------
# Brief information
# ---------------------------------------------
print("Train shape:", train_df.shape)
print(train_df.head())
print("\nTest shape:", test_df.shape)
print(test_df.head())

# ---------------------------------------------
# There are 4 columns in train: [textID, text, selected_text, sentiment]
# 'sentiment' is the label (positive, neutral, negative).
# The 'text' column is the entire tweet, while 'selected_text'
# is the phrase within the text that best represents the sentiment.
# For sentiment analysis, we will use the 'text' or 'selected_text' column.
# ---------------------------------------------

# =============================================
# 2) SPLIT TRAINING & VALIDATION DATA
# =============================================
# We do not have labels for test.csv, so we need
# to split the training data into train & validation sets
# so we can measure the performance of our model.
# ---------------------------------------------
X = train_df['text']           # Or you can use 'selected_text' if preferred
y = train_df['sentiment']

# Split data (e.g., 80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================
# PREPROCESSING DATA
# =============================================
def preprocess_text(text):
    # Clean the text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing
X_train = X_train.apply(preprocess_text)
X_val = X_val.apply(preprocess_text)

# =============================================
# 3) BUILD PIPELINE & TRAIN MODEL
# =============================================
# Example pipeline consists of:
# - TfidfVectorizer: converts text into TF-IDF vectors
# - LogisticRegression: a simple classification model
# ---------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",  # remove common words
        ngram_range=(1,2),     # use unigram & bigram
        max_features=5000      # limit the number of features
    )),
    ("clf", LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        random_state=42
    ))
])

# Hyperparameter tuning
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__max_features": [5000, 10000],
    "clf__C": [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Use the best model
best_model = grid_search.best_estimator_

# =============================================
# 4) EVALUATE USING F1 METRIC (MACRO)
# =============================================
# Since F1 on multiclass data -> Macro F1
# ---------------------------------------------
y_val_pred = best_model.predict(X_val)
f1_macro = f1_score(y_val, y_val_pred, average='macro')

print(f"\nF1 macro after tuning: {f1_macro:.4f}")
print("Best parameters:", grid_search.best_params_)

# =============================================
# 5) PREDICT ON TEST DATA
# =============================================
# The test.csv data does not have a 'sentiment' column -> labels are not provided
# ---------------------------------------------
X_test = test_df['text']  # or 'selected_text'
test_predictions = best_model.predict(X_test)

# Example: display the first 10 predictions
print("\nExample of 10 sentiment predictions for test data:")
print(test_predictions[:10])

# =============================================
# OPTIONAL: SAVE PREDICTIONS
# =============================================
# If you want to create a submission file (example format):
# ---------------------------------------------
submission_df = pd.DataFrame({
    'textID': test_df['textID'],
    'text': test_df['text'],
    'selected_text': test_df['selected_text'],
    'predicted_sentiment': test_predictions
})

submission_df.to_csv("my_submission.csv", index=False)
print("\nFile 'my_submission.csv' has been saved.")