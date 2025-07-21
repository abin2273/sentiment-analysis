Sentiment Analysis of Tweets using SVM
This project demonstrates a complete machine learning pipeline for sentiment analysis on tweet data, from training to prediction. It classifies tweets as either negative or not negative. The process involves text preprocessing, feature extraction using TF-IDF, classification using a Support Vector Machine (SVM), and generating a submission file with predictions on a test set.

Project Structure
The script performs the following key steps:

Load Data: Reads the training tweet data from a CSV file. Includes a fallback to a sample DataFrame if the file isn't found.

Preprocess Text: Cleans the tweet text by converting it to lowercase, removing URLs and mentions, tokenizing, lemmatizing, and removing stopwords.

Split Data: Divides the dataset into training and validation sets for model evaluation.

Build & Train Pipeline: Constructs and trains a scikit-learn Pipeline that combines TfidfVectorizer for feature extraction and SVC for classification.

Evaluate Model: Assesses the trained model's performance on the validation set using metrics like the F1-score and a classification report.

Predict on Test Data: Loads a separate test dataset, applies the same preprocessing steps, and uses the trained pipeline to predict sentiment labels.

Create Submission File: Saves the predictions into a submission.csv file in the required format for a competition or further use.

Code
Here is the complete Python script that covers training, evaluation, and prediction.

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# --- 1. Load the Training Dataset ---
try:
    # Make sure to provide the correct path to your training data
    train_df = pd.read_csv('/content/drive/MyDrive/DATA/train_2kmZucJ.csv')
except FileNotFoundError:
    print("Training file not found. Using a sample DataFrame for demonstration.")
    train_df = pd.DataFrame({
        'id': range(3),
        'label': [0, 1, 0],
        'tweet': ['I love my new phone!', 'My laptop is so slow and buggy $&@*#', 'Just got the new headset, amazing quality.']
    })

# --- 2. Preprocessing Setup ---
# Download necessary NLTK data for tokenization, stopwords, and lemmatization.
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the lemmatizer and define English stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    - Converts to lowercase
    - Removes URLs, mentions, and non-alphanumeric characters (keeps $, &, *, #)
    - Tokenizes, lemmatizes, and removes stopwords
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'[^\w\s\$&@\*#]', '', text) # Remove punctuation except for some symbols
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned_tokens)

# Apply the preprocessing function to the 'tweet' column
print("Preprocessing text data...")
train_df['cleaned_tweet'] = train_df['tweet'].apply(preprocess_text)
print("Preprocessing complete.")

# --- 3. Define Features and Target & Split Data ---
X = train_df['cleaned_tweet']
y = train_df['label']

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 4. Build and Train the SVM Pipeline ---
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
    ('svc', SVC(kernel='linear', class_weight='balanced', random_state=42, probability=True))
])

print("\nTraining the SVM model...")
svm_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate the SVM Model ---
y_pred_svm = svm_pipeline.predict(X_val)
val_f1_svm = f1_score(y_val, y_pred_svm, average='weighted')

print("\n--- SVM Model Evaluation ---")
print(f"Validation Weighted F1-Score: {val_f1_svm:.4f}\n")
print("Validation Classification Report:")
print(classification_report(y_val, y_pred_svm, target_names=['Not Negative (0)', 'Negative (1)']))


# --- 6. Predict on the Test Data ---
print("\nLoading and preprocessing test data...")
try:
    # Make sure to provide the correct path to your test data
    test_df = pd.read_csv('/content/drive/MyDrive/DATA/test_oJQbWVk.csv')
except FileNotFoundError:
    print("Test file not found. Using a sample DataFrame for demonstration.")
    test_df = pd.DataFrame({
        'id': range(2),
        'tweet': ['This computer is a piece of junk.', 'The camera on this tablet is fantastic.']
    })

test_df['cleaned_tweet'] = test_df['tweet'].apply(preprocess_text)
print("Test data preprocessing complete.")

print("\nMaking predictions on the test set...")
X_test = test_df['cleaned_tweet']
test_predictions = svm_pipeline.predict(X_test)


# --- 7. Create and Save the Submission File ---
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': test_predictions
})

submission_df.to_csv('submission_svm.csv', index=False)

print("\nSubmission file 'submission_svm.csv' created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())

How to Run
Install Libraries: Make sure you have the required Python libraries installed.

pip install pandas numpy scikit-learn nltk

Datasets:

Download the training data (train_2kmZucJ.csv).

Download the test data (test_oJQbWVk.csv).

Place both files in the correct directory or update the file paths in the script.

Execute: Run the Python script. It will perform all steps from training to creating the final submission_svm.csv file.
