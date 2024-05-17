# Tweets Naive Bayes Classifier

## Description
This project is a Naive Bayes classifier implemented in Python to categorize tweets. It uses machine learning techniques to analyze text and determine the sentiment of tweets as positive, negative, or neutral. This classifier can be useful in social media analytics, marketing analysis, and more.

## Features
- **Tweet Sentiment Analysis**: Classify tweets into positive, negative, or neutral categories.
- **Data Preprocessing**: Includes scripts for cleaning and preparing tweet data for analysis.
- **Model Training and Evaluation**: Scripts to train the Naive Bayes model and evaluate its accuracy.

## How we did it 

1. We started with a dataset of labeled tweets where the labels are 0 and 1, 0 referf to normal tweets and one refers to harmful tweets.
2. We started with cleaning the dataset where we cleaned the database from unwanted straings such as mentions, hashtags, number, emojis, and etc.., where you can find all of the dataset related script [HERE](manage_dataset/)
3. After cleaning the dataset, we starting with spliting the dataset according to it's label, then with each labeled dataset, we started with splitting it into training and testing datasets, and at the we started with balancing the datasets, you can find all the related script to do each of these step [HERE](manage_dataset/) as well
4. after getting the dataset side done we started with the training phase, where we got vectorizer and classifier model as pkl models. You find the trained models [HERE](trained_models/), and you can find the training script here as well [HERE](train.py)
5. 
## Evaluation
- **Evaluation Report**


## Installation

To set up this project, you will need Python installed on your machine. You can clone this repository and install the required dependencies as follows:

```bash
git clone https://github.com/Jawabreh0/tweets-naive-baye.git
cd tweets-naive-bayes
pip install -r requirements.txt
```

