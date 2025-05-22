import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load training data
train_data = pd.read_csv("dataset/train.csv")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data["text"], train_data["sentiment"], test_size=0.2, random_state=42)

# Preprocess text
def preprocess_text(text):
    # simple text cleaning
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

X_train = X_train.apply(preprocess_text)
X_val = X_val.apply(preprocess_text)

# Build and train the model
model = Pipeline([
    ("vect", TfidfVectorizer()),
    ("clf", LogisticRegression())
])
model.fit(X_train, y_train)

# Evaluate the model
preds = model.predict(X_val)
print("Validation Macro F1-Score:", f1_score(y_val, preds, average="macro"))

# Load test data
test_data = pd.read_csv("test.csv")

# Predict sentiment for the test set
test_preds = model.predict(test_data["text"])
print("Predictions for test set:", test_preds)