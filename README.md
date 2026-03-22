# 🗂️ Automated Credit Card Fraud Detection

**By Rashid Iqbal | AI Red Teamer 🛡️**  
**Cyber Internship Project @ ARCH Technologies**

A Python-based machine learning system that detects fraudulent credit card transactions by analyzing anomalies in transactional patterns. Unlike conventional rule-based approaches, this system uses statistical and machine learning techniques to identify suspicious activities in real-time.

---

## 🎯 Objective

1. Detect fraudulent credit card transactions using machine learning models 💳  
2. Learn anomaly detection techniques: Isolation Forest, Local Outlier Factor, One-Class SVM 🔍  
3. Perform exploratory data analysis to understand feature correlations and class distribution 📊  
4. Gain hands-on experience in data preprocessing, model training, and evaluation for cybersecurity applications 🛡️  

---

## 🧠 Key Concepts

### Data Exploration & Visualization
- Loaded dataset using `pandas` and inspected structure 🐼  
- Checked for missing values; handled highly imbalanced data  
- Visualized transaction distributions for fraud vs. normal cases using histograms and log-scaled axes 📈  
- Generated heatmap of feature correlations to understand relationships 🔎  

### Sampling & Feature Selection
- Reduced dataset size using random sampling (`frac=0.1`) for faster experimentation  
- Separated target column (`Class`) from input features  
- Converted data into feature matrix `X` and target vector `y`  

### Anomaly Detection Models
- **Isolation Forest**: isolates anomalies using random forest structures 🌲  
- **Local Outlier Factor (LOF)**: detects outliers based on local density deviations 📏  
- **One-Class SVM**: learns boundary of normal transactions and flags deviations as fraud 🛡️  

### Model Training & Prediction
- Configured models with proper hyperparameters and contamination rates  
- Trained each model on the sampled dataset  
- Converted predictions to consistent format: `0 = Normal`, `1 = Fraud`  
- Calculated errors, accuracy, and classification reports for evaluation  

### Evaluation
- **Isolation Forest**: Accuracy ≈ 99.78%, detected several fraudulent transactions  
- **Local Outlier Factor**: Accuracy ≈ 99.68%, lower recall due to imbalance  
- **One-Class SVM**: Performance depended on kernel and hyperparameter selection  
- Evaluated models using precision, recall, F1-score, and support metrics  

### Data Handling
- Used `numpy` and `pandas` for numerical and tabular data manipulation  
- Visualized distributions and correlations using `matplotlib` and `seaborn`  
- Ensured reproducibility using a fixed random seed  

---

## 🐍 Python Implementation

- Imported essential libraries: `numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`  
- Loaded dataset with `pd.read_csv()` and inspected with `data.info()`  
- Split dataset into fraud and normal transactions for analysis  
- Calculated outlier fraction to parameterize anomaly detection models  
- Defined models in a dictionary and iterated to train & evaluate each  
- Used loops to handle predictions and compute performance metrics  
- Visualized class distributions, transaction amounts, and feature correlations  

---

## 💡 Practical Use

- **Real-Time Fraud Detection**: Detect fraudulent transactions as they occur, helping banks prevent losses 💳  
- **Handling Imbalanced Data**: Demonstrates strategies for highly skewed datasets common in financial fraud ⚖️  
- **Risk Management & Decision Support**: Flags suspicious transactions for further investigation and supports automated alerts 🚨  
- **Data-Driven Cybersecurity Analytics**: Strengthens skills in preprocessing, visualization, and ML model evaluation 📊  
- **Scalable Methodology**: Framework can handle millions of transactions, relevant for fintech & banking systems 🌐  

---

## ✅ Conclusion

- Built a machine learning system for credit card fraud detection 💳  
- Gained practical experience in anomaly detection and imbalanced datasets  
- Learned data preprocessing, visualization, and model evaluation techniques 🛠️  
- Strengthened Python and machine learning skills for real-world cybersecurity applications 🛡️  

---

## 👨‍💻 Author

**Rashid Iqbal**  
**GitHub:** https://github.com/NoxVesper  
**📧 Email:** echoinject@gmail.com  

Suggestions & feedback welcome! 📝  

---

⚠️ **Disclaimer:** Educational use only. Use only on datasets you own or have permission to analyze.
