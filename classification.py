import pickle
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the trained model and vectorizer
logger.info("Loading the trained model and TF-IDF vectorizer...")
with open('./trained_models/naive_bayes_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)
with open('./trained_models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
logger.info("Model and vectorizer loaded successfully.")

def classify_tweet(tweet):
    """
    Classify a tweet as 'normal' or 'harmful'.

    Parameters:
    tweet (str): The tweet to classify.

    Returns:
    str: The classification result ('normal' or 'harmful').
    """
    # Convert the tweet to numerical features using the TF-IDF vectorizer
    logger.info("Converting the tweet to numerical features using the TF-IDF vectorizer...")
    tweet_tfidf = vectorizer.transform([tweet])
    
    # Predict the class using the Naive Bayes classifier
    logger.info("Classifying the tweet...")
    prediction = clf.predict(tweet_tfidf)
    
    # Map the prediction to the corresponding label
    result = 'harmful' if prediction[0] == 1 else 'normal'
    logger.info(f"The tweet has been classified as {result}.")
    
    return result

if __name__ == "__main__":
    # Example usage
    example_tweet = "Fuck you i hate you"
    result = classify_tweet(example_tweet)
    print(f"The tweet is classified as: {result}")
