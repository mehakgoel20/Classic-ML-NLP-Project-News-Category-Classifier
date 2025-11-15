# Classic-ML-NLP-Project-News-Category-Classifier
This project builds a news category classifier using classical NLP + ML techniques on a custom-engineered BBC News dataset. Starting with 42,000 raw RSS articles, I cleaned, deduplicated, and labeled the dataset using a custom URL-based category extraction pipeline, resulting in a high-quality 33,000-sample dataset across 8 categories.


📰 Project Overview

The final model uses TF-IDF + Linear SVM, achieving 86.6% accuracy, further improved through hyperparameter tuning with GridSearchCV.

🚀 Key Features

✔️ Engineered a 42k → 33k curated BBC dataset
✔️ Built custom URL-based label extraction (world, politics, sport, business, etc.)
✔️ Text preprocessing using tokenization, cleaning, stopword removal, lemmatization (spaCy)
✔️ Implemented and compared Naive Bayes, Logistic Regression, Linear SVM
✔️ Achieved 86.6% test accuracy with LinearSVC
✔️ Applied GridSearchCV for hyperparameter tuning
✔️ Visualized performance using confusion matrix
✔️ Extracted top TF-IDF features for interpretability
✔️ Built a scalable ML pipeline ready for deployment

📂 Dataset Description

Source: BBC RSS feeds
Initial size: 42,000 articles
Final size: 33,000 cleaned and labeled samples

Columns used:
title
description
guid (link used for category extraction)
final_category (engineered label)
text (title + description merged + lemmatized)

Categories (after cleaning):

world
politics
business
sport
entertainment
technology
health
science

🧹 Data Cleaning & Preprocessing

Removed duplicates and invalid rows
Dropped rows with missing category labels (None/misc)
Cleaned text: lowercasing, punctuation removal
Lemmatized using spaCy
Combined title + description into unified input text
🔧 Modeling Approach

1️⃣ TF-IDF Vectorization

stop_words = "english"
max_features = 50,000
(1,1) and (1,2) n-grams tested

2️⃣ Models Compared

Model	Accuracy
Naive Bayes	~76%
Logistic Regression	~84%
Linear SVM	86.6%
3️⃣ GridSearchCV Tuning

Parameters tuned:

tfidf__max_features
tfidf__ngram_range
tfidf__min_df
tfidf__max_df
clf__C
clf__class_weight
Improved consistency and category-wise F1 performance.

📊 Results

✔️ Final Accuracy: 86.6%
✔️ Best Model: LinearSVC
✔️ Insights generated:
Confusion matrix heatmap
Top keywords per category (TF-IDF feature weights)
