# 🎵 Large-Scale Sentiment Analysis Using Apache Spark

This project is a comprehensive sentiment analysis pipeline built on Apache Spark using the **Amazon "CDs and Vinyl" Review Dataset**. The goal was to classify reviews into **positive or negative sentiment** using large-scale distributed computing techniques. The work was completed in a group as part of the Applied Big Data and Visualisation module at the University of Limerick.

---

## 📘 Project Overview

Millions of product reviews on Amazon hold valuable customer feedback. We utilized **Apache Spark and PySpark MLlib** on **Databricks Community Edition** to analyze and classify these reviews. Our focus: build a scalable and accurate machine learning pipeline for sentiment analysis.

---
## 📁 Dataset: Amazon CD/Vinyl Reviews

- **Source**: [Amazon Customer Review Dataset](https://nijianmo.github.io/amazon/index.html)
- **Subset Used**: `CDs and Vinyl`
- **Format**: JSON
- **Records**: ~1.2 million reviews

---
## 🧠 Key Skills Demonstrated

### 🔹 Big Data Processing
- Used **Apache Spark** for distributed data handling of 1.2M+ reviews.
- Implemented **Spark DataFrames**, **SparkSQL**, and **DBFS storage**.
  
### 🔹 Data Preprocessing & Cleaning
- Removed duplicates and NULLs
- Converted and normalized fields (dates, helpful votes, etc.)
- Standardized and tokenized textual data (review text)

### 🔹 Exploratory Data Analysis (EDA)
- Visualized trends using **Matplotlib** and **Seaborn**
- Temporal, categorical, and helpfulness-based insights
- Identified most common product styles and review behavior over time

### 🔹 Feature Engineering
- TF-IDF vectorization for converting review text into numerical format
- Pearson correlation analysis
- Feature normalization via Standard Scaler

### 🔹 Machine Learning
- Binary sentiment classification using **Logistic Regression**
- Built full ML pipeline: TF-IDF → Train/Test Split → Model Training
- Evaluated using **Accuracy**, **F1 Score**, **Confusion Matrix**, **ROC Curve**

### 🔹 Hyperparameter Tuning
- Used **Cross-Validation** with ElasticNet and Regularization parameters
- Achieved high performance:  
  - Final Accuracy: **93.68%**  
  - Final F1 Score: **91.74%**


---

## 🔬 Technologies Used

- **Apache Spark** (PySpark MLlib)
- **Databricks Community Edition**
- **Python 3**
- **Matplotlib & Seaborn** (for visualizations)
- **Pandas / SparkSQL**
- **TF-IDF, Logistic Regression, CrossValidator**

---

## 📈 Results & Insights

- The dataset is **heavily skewed** toward positive reviews.
- The model struggles slightly with **negative class detection**, a common challenge in imbalanced datasets.
- Most active review period: **2011–2016**
- Reviews with low star ratings received **higher helpfulness votes**, indicating useful critical feedback.

---

## 🚧 Challenges & Limitations

- **Databricks cluster timeouts** disrupted workflow (resolved with caching and efficient pipeline design).
- **3-star reviews were excluded** to maintain binary classification simplicity, but this reduced nuance.
- Negative review classification remains less accurate — future improvements could include ensemble models or NLP fine-tuning.

---

## 🚀 Future Work

- Expand to **multi-class sentiment** (positive, neutral, negative)
- Real-time sentiment classification using **Spark Streaming**
- Build an interactive **dashboard** with live visualizations
- Experiment with **deep learning models** (e.g., BERT, LSTM)

---

## 📄 Project Report

📘 **[Download the Full Report (PDF)](https://1drv.ms/b/c/612d6f6621161847/EQlgdu0v3gFIut80xuFSUOIBgL0mp9GaPR9q5PB7QfcMBA?e=RlVAGP)**  
Includes all code stages, visualizations, results, and detailed explanation of the methodology.

