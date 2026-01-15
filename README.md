# Naive Bayes Fake News Classification (From Scratch)

## Overview
This project implements a Naive Bayes text classifier **from scratch** to distinguish real and fake news articles using raw textual content.

Rather than relying on pre-built NLP models, the classifier is implemented manually using Python data structures and probability calculations. The project demonstrates a clear understanding of probabilistic modeling, text preprocessing, and evaluation metrics for binary classification in natural language processing.

---

## Problem Statement
Automatically distinguishing real news from fake news is a classic text classification problem that highlights both the power and limitations of simple probabilistic models.

The objective of this project is to:
- Build a Naive Bayes classifier without using high-level NLP APIs
- Learn how word probabilities drive document-level predictions
- Evaluate classification performance using precision, recall, and F1 score

This task is intended for educational purposes and does not attempt to solve the full real-world complexity of fake news detection.

---

## Dataset
- **Source**: Fake and Real News Dataset (Kaggle)
- **Size**: 44,898 news articles
- **Classes**:
  - Real news
  - Fake news
- **Features**:
  - News title
  - News text
  - Combined into a single document field

Each document is labeled as real or fake and used as input to the classifier.

---

## Data Preprocessing
The following preprocessing steps are applied:
- Combine title and text into a single document
- Convert all text to lowercase
- Tokenize text using regular-expression-based splitting
- Build vocabulary from the training data only
- Split data into training and test sets (80/20)

No external NLP libraries are used for tokenization or feature extraction.

---

## Methodology: Naive Bayes (Implemented by Hand)

### Model Assumptions
- Bag-of-words representation
- Conditional independence between words given the class
- Multinomial Naive Bayes formulation

### Key Implementation Details
- Word frequencies are counted separately for real and fake news
- Class priors are computed from training data
- Conditional word probabilities are computed using **Laplace (add-1) smoothing**
- Log-probabilities are used to avoid numerical underflow
- Predictions are made by comparing posterior probabilities for each class

All probability calculations are explicitly implemented in Python without calling pre-built Naive Bayes classifiers.

---

## Evaluation Metrics
Model performance is evaluated on the held-out test set using:
- **Precision**
- **Recall**
- **F1 Score**

These metrics provide a balanced view of classification performance, particularly in the presence of class imbalance and asymmetric error costs.

---

## Results & Observations
- The Naive Bayes classifier achieves strong baseline performance on the dataset
- Fake news detection benefits from distinctive word usage patterns
- Errors often occur in articles with neutral or ambiguous language
- Despite its simplicity, Naive Bayes performs competitively for large-scale text classification

This highlights why Naive Bayes remains a strong baseline model for many NLP tasks.

---

## Limitations
- Bag-of-words representation ignores word order and semantics
- Independence assumption is unrealistic for natural language
- Model is sensitive to vocabulary choice and preprocessing decisions

More advanced approaches (e.g., TF-IDF, word embeddings, transformer models) could address these limitations but are intentionally excluded here.

---

## Project Structure
├── Naive_Bayes_Classification.ipynb   # Main implementation notebook
├── data/                             # Raw CSV files (Fake.csv, True.csv)
└── README.md
---

## Tools & Technologies
- Python
- Pandas
- Regular expressions (re)
- NumPy
- Scikit-learn (metrics only)
- Jupyter Notebook

---

## Key Takeaways
- Implemented Naive Bayes classification entirely from first principles
- Gained hands-on understanding of probabilistic text modeling
- Demonstrated how simple models can scale effectively to large text datasets
- Built a strong NLP baseline without relying on black-box libraries

---

## License
This project is for academic and educational demonstration purposes.
