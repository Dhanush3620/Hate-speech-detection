
## Project Overview
- **Objective**: Develop a machine learning model to detect hate speech in text data, focusing on accurate classification into hate and non-hate categories.

 
This project focuses on Hate Speech Detection using Natural Language Processing (NLP) techniques and machine learning models to identify hateful text on social media. Motivated by the increasing prevalence of hate speech, the dataset reflects social media trends, including emoticons and slang, which complicate detection. It is preprocessed and categorized into hateful ("1") or non-hateful ("0") text, enabling the training of effective models. This benchmark dataset supports Deep Learning (DL) and NLP applications, aiding in the development of automated systems to filter harmful content while adhering to policy guidelines to reduce cyber harm.

---
 
 ## A: Project Idea
 Researched various project ideas, discussed with group members, and collectively decided on an NLP-based project focused on Hate Speech Detection.
 
 ## B: Dataset Curation and Preprocessing
Removed duplicates and null values, replaced them with correct data, checked that the context contained only text and no extraneous characters. Used the ‘langdetect‘ library to ensure only English sentences were retained in the dataset.
 
 ## C: Data Exploration and Summary Statistics
 Identified critical tests for the project, performed hypothesis testing, and plotted results to evaluate the null hypothesis, which was rejected. Illustrated findings with box plots, conducted Chi-Square tests, and analyzed data distribution.
 
 ## D: ML Algorithm Design/Development
Analyzed the data thoroughly and chose two machine learning models: the DistilBERT model and the LSTM model. For DistilBERT, used the pre-trained ‘distilbert-base-uncased‘ model and split the data into 80 percent training and 20 percent testing. Performed tokenization for both datasets and created data loaders with a batch size of 16. Se lected the AdamW optimizer for training. For the LSTM model, tokenized the data and selected ReLU and Sigmoid activation functions. Used the Adam optimizer to optimize model performance.
 
Naive Bayes, trained with TF-IDF vectorization and an 80/20 train-test split, efficiently leveraged feature independence for classification, with per formance evaluated using precision, recall, F1-score, accuracy, confusion matrices, and visualized
 through bar charts.
 
 ## E: ML Algorithm Training and Test Data Analysis
 
 • DistilBERT: Completed tokenization, created data loaders, and trained the model for three epochs using the AdamW optimizer. Printed and plotted loss for each epoch.
 
 • LSTM: Split data into 80 percent training and 20 percent testing. Trained for five epochs and calculated accuracy and loss for evaluation.
 
 
 • Naive Bayes: The dataset was vectorized using TF-IDF, split into 80 percent training and 20 percent testing, and trained using the Multinomial Naive Bayes algorithm, well-suited for text classification.
 
 ## F: Visualization, Result Analysis, Conclusion
Visualized accuracy metrics and evaluation matrices such as confusion matrices and AUROC curves. Compared models and concluded that DistilBERT performed best.
 
 ## G: Final Tutorial Report Creation
Prepared the final project report and created the GitHub Pages tutorial. Also contributed to model selection and training.
 
---

## Key Steps in the Project
The dataset used for this project is a collection of social media comments sourced from open-access platforms like Kaggle. It contains text labeled as either hateful ("1") or non-hateful ("0"). The hateful category includes offensive language, while the non-hateful category contains neutral comments.

1.**Data preparation** involved several steps:

Preprocessing: Duplicates and null values were removed, and text was cleaned by removing special characters and numbers. Only English comments were retained using the langdetect library. The labels were verified for accuracy.
Transformation: The cleaned data was stored in a pandas DataFrame with two columns: "Text" and "Label." Text was tokenized and padded to a uniform length of 100 tokens.
Storage: The processed dataset was stored as a CSV file, enabling efficient querying and machine learning integration.
These steps were essential to ensure the dataset was clean, consistent, and ready for analysis, which is vital for building accurate machine learning models.

**Exploratory Data Analysis (EDA)**


Exploratory Data Analysis (EDA) was conducted to gain insights into the dataset, including its structure, distribution, and key characteristics.

Label Distribution: The dataset consists of comments labeled as either hateful ("1") or non-hateful ("0"). The EDA revealed an imbalance, with more non-hateful comments than hateful ones. This imbalance was addressed during the machine learning process.

**Statistical Tests and Hypothesis Testing**:

Hypothesis testing was performed to examine if there was a significant difference in the frequency of hateful and non-hateful comments. The null hypothesis was rejected based on the results.

A Chi-Square test was performed to analyze the relationship between specific words and their association with hateful or non-hateful labels, confirming the statistical significance of certain terms in identifying hate speech.

Distribution Analysis: The length of comments (in terms of word count) was analyzed, showing that hateful comments tend to be shorter than non-hateful ones. A box plot was created to visualize this distribution, highlighting the difference in comment lengths between the two classes.

Langdetect was utilized to filter the dataset, retaining only English sentences to ensure relevance and consistency for the hate speech detection task. Each sentence was analyzed using the library, and only those identified as English and containing more than three words were kept. This preprocessing step removed non-English and overly short sentences, improving the dataset's quality for further analysis.

![image](https://github.com/user-attachments/assets/0867228f-ef87-4941-b9b8-692cdf4d9e8a)

**Key Visualizations**:

Bar plots showing the frequency of hateful vs. non-hateful comments.
A word cloud to visualize the most common words in each category (hateful and non-hateful comments).
A box plot illustrating the distribution of comment lengths across the two labels.
Findings and Insights:
Hateful comments often include specific keywords or slang, while non-hateful comments tend to be longer and more formal.
The dataset’s imbalance required further preprocessing to ensure fair model training.
Conclusion:
The EDA provided critical insights into the dataset, allowing for informed decisions in the subsequent stages of the project. By understanding the dataset’s characteristics, the preprocessing and modeling approaches were tailored to effectively address the challenges of hate speech detection.

## Primary/Machine Learning Analysis

Based on the insights from the Exploratory Data Analysis (EDA), three machine learning techniques were selected for the project: **DistilBERT**, **LSTM (Long Short-Term Memory)**, and **Naive Bayes**. These models were chosen for their effectiveness in processing and analyzing textual data, especially for Natural Language Processing (NLP) tasks such as hate speech detection.

### Why were these techniques chosen?

- **DistilBERT**: A smaller and faster version of the BERT (Bidirectional Encoder Representations from Transformers) model. It is pre-trained on large corpora and excels at understanding the context of words in sentences. DistilBERT’s transfer learning capability allows it to adapt quickly to hate speech detection with minimal computational resources.

- **LSTM**: A type of Recurrent Neural Network (RNN) effective for processing sequential data like text. It captures long-term dependencies, crucial for understanding the context of words and phrases, making it ideal for identifying subtle cues in language.

- **Naive Bayes**: A computationally efficient and straightforward model. It assumes independence between features and is useful for quick text analysis, providing a solid baseline for comparison.

### How do these techniques help answer the key questions?

- **DistilBERT**: By tokenizing and transforming text into embeddings, DistilBERT captures word relationships and context. It focuses on the critical parts of the input, ensuring accurate classification of hateful versus non-hateful comments.
  
- **LSTM**: Processes tokenized data and identifies sequential patterns, detecting relationships between words over time. This model can distinguish subtle language cues that separate hate speech from non-hateful content.
  
- **Naive Bayes**: Offers quick training and prediction times, ideal for initial exploration of text data distributions and feature importance, especially through TF-IDF.

### Implementation Details:

- **DistilBERT**: Utilized the pre-trained `distilbert-base-uncased` model. Text data was tokenized with the DistilBERT tokenizer, and data loaders were created with a batch size of 16. The AdamW optimizer was chosen for training, and the model was fine-tuned for 3 epochs.
  
- **LSTM**: The text data was tokenized and padded to ensure uniform input length. The model used ReLU and Sigmoid activation functions with the Adam optimizer. The LSTM model was trained for 5 epochs, with accuracy and loss monitored during training.
  
- **Naive Bayes**: The dataset was vectorized using the TF-IDF approach. The data was split into 80% training and 20% testing sets. The Multinomial Naive Bayes variant was used, providing an efficient but less nuanced approach compared to DistilBERT and LSTM.

### Conclusion:

- **DistilBERT** and **LSTM** were highly effective in addressing the project’s objectives. DistilBERT excelled due to its advanced contextual understanding, while LSTM provided a complementary approach by analyzing sequential relationships in text.
  
- **Naive Bayes**, while efficient and useful for initial exploration, struggled with more complex patterns in language. It was still valuable for understanding feature importance and text distributions but was outperformed by DistilBERT and LSTM in detecting nuanced language patterns.

## Visualization

In this section, we present key visualizations that provide insights into the analysis, preprocessing, and performance of the hate speech detection models.

1. **Text Length Distribution by Label**  
   This plot shows the distribution of the number of words in texts labeled as **Hate Speech** and **Non-Hate Speech**, indicating that **Non-Hate Speech** tends to have longer text lengths.
   ![image](https://github.com/user-attachments/assets/951b7322-9a36-41d4-b5b5-d3fc2cfe1064)
 
                                             **Figure 1**: Text Length Distribution by Label

2. **Cumulative Distribution of Text Lengths by Label**  
   This plot illustrates the cumulative percentage of text lengths, showing that shorter text lengths dominate in both categories.  

![image (1)](https://github.com/user-attachments/assets/0de1a08e-2305-4f0b-9a4d-d7e4d36ad0bf)

                                            **Figure 2**: Cumulative Distribution of Text Lengths by Label

3. **Mean Sentiment Score by Speech Type**  
   This plot compares the **mean sentiment scores** for **Hate Speech** and **Non-Hate Speech**, with error bars indicating statistical significance.  
![image (2)](https://github.com/user-attachments/assets/e9424115-e9d4-4f4d-ba85-997bec6a340b)

                                  **Figure 3**: Mean Sentiment Score by Speech Type with Error Bars

4. **Contingency Table: Label vs Sentiment Category**  
   This heatmap displays the relationship between text labels and sentiment categories, highlighting significant differences across categories.  
![image (3)](https://github.com/user-attachments/assets/1d8a0839-df55-44d2-b197-b471b0b0b7ce)

                                  **Figure 4**: Contingency Table: Label vs Sentiment Category

5. **Distribution of Sentiment Scores**  
   This histogram shows the **distribution of sentiment scores** for both **Hate Speech** and **Non-Hate Speech**, emphasizing patterns in sentiment polarity.  
![image (4)](https://github.com/user-attachments/assets/8e0c4fa8-c08a-4bc5-a27f-c940a3a128b8)

                                 **Figure 5**: Distribution of Sentiment Scores: Hate Speech vs Non-Hate Speech

6. **Model Performance Comparison**  
   This plot compares the performance metrics (Accuracy, Precision, Recall, F1-Score) of **Naive Bayes**, **LSTM**, and **DistilBERT**, with **DistilBERT** emerging as the best model.  
![image (5)](https://github.com/user-attachments/assets/600fd553-2771-4380-b4c0-e6ebba9b7fff)

                                          **Figure 6**: Model Performance Comparison



## Insights and Conclusions

This project focused on **hate speech detection** using advanced **Natural Language Processing (NLP)** techniques, specifically **DistilBERT**, **LSTM**, and **Naive Bayes** models. By combining modern machine learning methodologies with robust data preprocessing and analysis, the project successfully addressed the challenges posed by real-world social media text. These efforts demonstrate how automation can play a vital role in moderating online content and promoting safer digital spaces.

### For an Uninformed Reader:
This project provides a clear understanding of the problem of hate speech detection and the steps involved in solving it. An uninformed reader gains insights into:
- The significance of identifying and addressing hate speech to combat online abuse and foster inclusive communication.
- How datasets are prepared for analysis, including data cleaning, filtering, and preprocessing to ensure quality inputs for machine learning models.
- The importance of choosing effective machine learning models, such as **DistilBERT**, **LSTM**, and **Naive Bayes**, to handle the complexities of textual data.

Through detailed explanations and visualizations, the report equips an uninformed reader with foundational knowledge about hate speech detection using machine learning.

### For a Reader Familiar with the Topic:
A knowledgeable reader gains valuable insights from this project, particularly in comparing the effectiveness of different NLP models. Key takeaways include:
- The comparative analysis of **DistilBERT**, **LSTM**, and **Naive Bayes** models, highlighting their respective strengths in handling text classification tasks.
- Strategies for addressing challenges such as data imbalance, ensuring fair and unbiased model training.
- The detailed evaluation of models using performance metrics like **accuracy**, **loss trends**, **confusion matrices**, and **AUROC curves**, which provide a nuanced understanding of their capabilities.

These insights offer advanced readers a practical perspective on implementing and evaluating machine learning models for hate speech detection.

### Key Conclusions:
- **DistilBERT** proved to be the most effective model, demonstrating superior contextual understanding and accuracy compared to **LSTM** and **Naive Bayes**.
- The incorporation of advanced preprocessing techniques, such as **language filtering** and **sequence padding**, further enhanced model performance.
- This project emphasizes that combining state-of-the-art NLP models with rigorous preprocessing is essential for tackling hate speech detection challenges.

## Data Science Ethics

The ethical considerations in this project focused on ensuring fairness, transparency, and the mitigation of biases in hate speech detection, given its sensitive nature.

### Potential Biases in Data Collection:
The dataset, sourced from open-access platforms, consisted of social media text labeled as hateful or non-hateful. Concerns included potential biases in the labeling process due to subjective human judgments, as well as societal biases in the data, such as overrepresentation of certain demographics or linguistic styles.

### Mitigation Strategies for Bias:
To address these concerns:
- The dataset was reviewed for label balance between hateful (1) and non-hateful (0) comments.
- Preprocessing steps, including duplicate removal and language filtering using the `langdetect` library, were applied to eliminate irrelevant data.
- The diversity of text inputs, including slang and emoticons, was maintained to ensure the model's generalization capability.

### Fairness in Model Development:
Machine learning models (DistilBERT, LSTM, and Naive Bayes) were chosen for their ability to handle diverse text data and imbalanced datasets. Model performance was monitored through accuracy, loss, and confusion matrices to ensure unbiased predictions and avoid overfitting or underfitting.

### Transparency in Analysis:
All preprocessing steps, model parameters, and evaluation criteria were fully documented in the report and GitHub repository. Visualizations like word clouds and AUROC curves were included to provide interpretable results, ensuring the methodology is reproducible and transparent.

### Final Thoughts:
This project bridges theory and practice in hate speech detection, offering a machine learning framework that tackles social media challenges while promoting safer, more inclusive digital spaces.

By addressing data biases and ensuring fairness and transparency in model development, the project upholds ethical principles in data science. The steps taken to mitigate ethical concerns not only enhance the credibility of the hate speech detection system but also contribute to creating more equitable and trustworthy AI solutions for societal challenges.

---

## License
[Include license information.]
