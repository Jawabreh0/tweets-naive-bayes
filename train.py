import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("Loading datasets...")
# Load the datasets
tweets_normal = pd.read_csv('./dataset/training_dataset/normal.csv')
tweets_harmful = pd.read_csv('./dataset/training_dataset/harmful.csv')

logger.info("Datasets loaded successfully.")

# Add a label column
tweets_normal['label'] = 0  # Normal tweets
tweets_harmful['label'] = 1  # Harmful tweets

# Combine the datasets
logger.info("Combining datasets...")
tweets = pd.concat([tweets_normal, tweets_harmful], ignore_index=True)
logger.info("Datasets combined successfully.")

# Extract features and labels
X = tweets['tweet']  # Assuming the column with tweets is named 'tweet'
y = tweets['label']

# Convert text data into numerical features using TF-IDF
logger.info("Converting text data into numerical features using TF-IDF...")
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(tqdm(X, desc="TF-IDF Vectorization"))

logger.info("TF-IDF vectorization completed.")

# Train the Naive Bayes classifier
logger.info("Training the Naive Bayes classifier...")
clf = MultinomialNB()
clf.fit(X_tfidf, y)

logger.info("Naive Bayes classifier training completed.")

# Save the trained model and vectorizer to disk
logger.info("Saving the trained model and vectorizer to disk...")
with open('./trained_models/naive_bayes_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
with open('./trained_models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

logger.info("Model and vectorizer saved to disk. Training process completed.")
