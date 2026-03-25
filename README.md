# Hate Speech Detection using NLP and Deep Learning

An end-to-end natural language processing project for detecting hate speech in social media text using **DistilBERT**, **LSTM**, and **Naive Bayes**.

## Overview

The goal of this project is to build and compare machine learning models for binary text classification, identifying whether a social media comment is **hateful** or **non-hateful**.

This project was designed as a complete NLP workflow rather than just a modeling exercise. It includes:

- data cleaning and language filtering
- exploratory data analysis and hypothesis testing
- text preprocessing and feature engineering
- model development using classical ML and deep learning
- performance comparison using multiple evaluation metrics
- result interpretation through visualizations

### Project Objective

This project answers the question:

> **How accurately can hate speech in social media text be detected using NLP and machine learning models, and which modeling approach performs best?**

## Dataset

The dataset consists of social media comments labeled as:

- **1** → Hate Speech
- **0** → Non-Hate Speech

The text data reflects real-world social media characteristics such as informal language, slang, short-form language, and noisy user-generated content, making it suitable for NLP classification tasks.

### Data Preparation

The dataset was prepared through several preprocessing steps:

- removed duplicates and null values
- cleaned text by removing special characters and numbers
- retained only English comments using the `langdetect` library
- verified and organized labels into a structured dataframe
- tokenized and padded text to a uniform length of 100 tokens for deep learning models
- prepared TF-IDF features for Naive Bayes

These steps improved data quality and ensured the dataset was consistent for downstream modeling.

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **Pandas / NumPy** | Data cleaning and manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Scikit-learn** | TF-IDF, Naive Bayes, evaluation metrics |
| **Transformers (Hugging Face)** | DistilBERT fine-tuning |
| **TensorFlow / Keras** | LSTM modeling |
| **langdetect** | Language filtering |
| **Jupyter Notebook** | Experimentation and analysis |
| **Git + GitHub** | Version control and project hosting |

## Project Workflow

### Phase 1 — Project Planning

- selected hate speech detection as the final NLP problem
- defined binary classification objectives and evaluation criteria
- structured the work around preprocessing, EDA, modeling, and reporting

### Phase 2 — Data Curation and Preprocessing

- cleaned the raw text data
- removed missing and duplicate records
- filtered non-English text using `langdetect`
- prepared labeled text for analysis and modeling

### Phase 3 — Exploratory Data Analysis

- analyzed class distribution between hate and non-hate labels
- examined text length patterns across classes
- performed hypothesis testing and Chi-Square analysis
- studied sentiment-related and distribution-based differences

### Phase 4 — Model Development

Built and compared three approaches:

- **DistilBERT** for contextual transformer-based classification
- **LSTM** for sequence-based deep learning classification
- **Naive Bayes** as a strong classical NLP baseline using TF-IDF

### Phase 5 — Model Training and Evaluation

- trained DistilBERT for 3 epochs using AdamW and batch size 16
- trained LSTM for 5 epochs using Adam with tokenized and padded sequences
- trained Multinomial Naive Bayes on TF-IDF features using an 80/20 train-test split
- evaluated models using accuracy, precision, recall, F1-score, confusion matrices, loss trends, and AUROC-based comparisons

### Phase 6 — Visualization and Reporting

- created visualizations for text characteristics and sentiment patterns
- compared model performance across evaluation metrics
- summarized findings in the final report and GitHub tutorial

## Key Modeling Notes

| Model | Key Details |
|------|-------------|
| **DistilBERT** | Used `distilbert-base-uncased`, tokenizer-based preprocessing, batch size 16, AdamW optimizer, and 3 training epochs |
| **LSTM** | Used tokenization and padding, ReLU and Sigmoid activations, Adam optimizer, and 5 training epochs |
| **Naive Bayes** | Used TF-IDF vectorization with an 80/20 train-test split and Multinomial Naive Bayes |

## Key Visual Analysis

The EDA and result analysis included visualizations such as:

- label distribution plots
- word frequency and word cloud analysis
- box plots for text length distribution
- cumulative text-length distribution
- contingency heatmaps for label vs sentiment category
- sentiment score distribution plots
- model performance comparison charts

These plots helped explain both the structure of the dataset and the comparative strengths of the models.

## Key Findings

### Dataset Characteristics

- the dataset was imbalanced, with more non-hateful comments than hateful ones
- hateful comments tended to be shorter than non-hateful comments
- English-only filtering improved consistency for modeling

### Model Performance

- **DistilBERT** delivered the strongest overall performance
- **LSTM** performed well by capturing sequential language patterns
- **Naive Bayes** served as a useful baseline but was less effective on nuanced text patterns

### Overall Conclusion

DistilBERT emerged as the best-performing model because of its stronger contextual understanding of language. The project showed that transformer-based approaches are especially effective for hate speech detection compared to traditional and sequence-based baselines.

## Why This Project Matters

Hate speech detection is an important NLP application for improving online safety and content moderation. This project demonstrates how machine learning can be applied to real-world text classification problems where language is noisy, subjective, and context-dependent.

Beyond model building, the project also highlights the importance of:

- robust preprocessing
- careful evaluation on imbalanced text data
- comparing baseline and advanced models
- interpreting results in a socially sensitive application domain

## Repository Structure

```bash
hate-speech-detection/
├── data/                     # raw and processed dataset files
├── notebooks/                # Jupyter notebooks for EDA, preprocessing, and modeling
├── models/                   # saved model artifacts
├── reports/                  # final report, tutorial, or presentation files
├── images/                   # plots and README visuals
├── requirements.txt          # project dependencies
└── README.md
```

## How to Run This Project

### Clone the Repo

```bash
git clone https://github.com/Dhanush3620/hate-speech-detection.git
cd hate-speech-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Analysis

- open the notebooks in Jupyter
- run preprocessing and EDA notebooks first
- run the model training notebooks/scripts for DistilBERT, LSTM, and Naive Bayes
- review generated plots and evaluation outputs

## Current Status and Future Improvements

### Completed

- data preprocessing and filtering
- exploratory data analysis
- statistical testing
- training and evaluation of DistilBERT, LSTM, and Naive Bayes
- visualization and final report creation

### Next Steps

- improve handling of class imbalance
- add cross-validation and hyperparameter tuning
- test additional transformer architectures
- deploy the best model as a simple web app or API
- add explainability methods for prediction interpretation

## Ethics and Responsible AI

Because hate speech detection is a socially sensitive task, ethical considerations are important.

Key concerns in this project include:

- bias in labeled social media data
- subjectivity in hate speech definitions
- risk of false positives or unfair moderation
- the need for transparent preprocessing and evaluation

This project attempts to address those concerns through documented preprocessing, multiple model comparisons, and transparent reporting of results.

## Author

**Dhanush Garikapati**  
M.S. in Data Science, University of Maryland
