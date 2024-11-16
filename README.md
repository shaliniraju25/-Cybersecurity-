# 🛡️ Microsoft: Classifying Cybersecurity Incidents with Machine Learning

---

## 📈 Problem Statement

Cybersecurity incidents are becoming more frequent and sophisticated. The goal of this project is to classify cybersecurity incidents into specific types using machine learning models. This classification helps enhance the efficiency of Security Operation Centers (SOCs) by enabling faster incident detection and response.

---
## 🎯 Objective:

- **Develop a machine learning model** to classify cybersecurity incidents based on a labeled dataset.
- **Optimize for imbalanced data** by applying data balancing techniques like SMOTE.
- **Deploy the model** using a Streamlit app for easy accessibility and real-time predictions.

---
## 🔍 Dataset Overview

- **Cybersecurity Incidents Dataset**: The dataset contains several features, such as incident categories, source IP, destination IP, protocols, timestamps, and other network-related data that capture incident patterns.
- **Target Variable**: `incident_category`, which includes categories such as malware, phishing, and network attacks.

---
## 💡 Business Use Cases

- **Incident Response**: Enable SOC teams to classify incidents quickly and respond effectively.
- **Threat Intelligence**: Identify trends and common attack types over time.
- **Automated Reporting**: Use predictions to generate real-time incident reports.

---
## 🛠️ Approach

### I. Data Preprocessing & Exploration

1. **Data Cleaning**:
   - Missing values were handled by filling or dropping columns based on thresholds.
   - Inconsistent entries were corrected, and features were standardized for optimal performance.
2. **Exploratory Data Analysis (EDA)**:
   - Investigated incident frequency, attack vectors, and correlations between features.
   - Visualized the top incident categories and trends over time.
     
![image](https://github.com/user-attachments/assets/01723739-dadd-4b80-baa3-2765ad1734cf)
![image](https://github.com/user-attachments/assets/f8a2229b-8faf-4fb3-9492-8bb06101ee3b)
![image](https://github.com/user-attachments/assets/02857e5a-e7a5-455c-9b6b-986fc45470a0)
![image](https://github.com/user-attachments/assets/39cf37e2-0ae1-42be-957d-4f599f5f2c16)
![image](https://github.com/user-attachments/assets/73d2a984-f04e-4ab7-812c-e733b3abda09)




### II. Data Balancing Techniques

To handle the class imbalance in cybersecurity incidents, **SMOTE (Synthetic Minority Over-sampling Technique)** was first applied. This ensured that underrepresented categories, such as 'malware' and 'phishing,' were better represented by generating synthetic samples for the minority classes.

## 🛠️ Applying SMOTE to the Training Data for Class Imbalance and Hyperparameter Tuning

Fitting 3 folds for each of 5 candidates, totaling 15 fits.

- **Best Hyperparameters**: 
  - `{'n_estimators': 75, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None, 'bootstrap': False}`

- **Classification Report**:
- | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | 0     | 0.69      | 0.58   | 0.63     | 765560  |
  | 1     | 0.40      | 0.58   | 0.47     | 390976  |
  | 2     | 0.71      | 0.65   | 0.68     | 628025  |
  | **Accuracy** |           |        | 0.60     | 1784561 |

## 🛠️ Improving Class Balance with SMOTE-ENN and Hyperparameter Tuning

While SMOTE provided a more balanced class distribution, further improvements were achieved by applying **SMOTE-ENN (SMOTE + Edited Nearest Neighbors)**:
- **SMOTE**: Generates synthetic samples for the minority class.
- **SMOTE-ENN**: First creates synthetic samples, then applies Edited Nearest Neighbors to remove noisy or ambiguous data points, enhancing the overall data quality.

Below, you can observe the class distribution before and after SMOTE-ENN:

**Class Distribution Before and After SMOTE-ENN**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.69      | 0.71   | 0.70     | 765560  |
| 1              | 0.50      | 0.54   | 0.52     | 390976  |
| 2              | 0.73      | 0.66   | 0.69     | 628025  |
| **Accuracy**   |           |        | 0.66     | 1784561 |

After addressing the class imbalance with SMOTE-ENN, hyperparameter tuning was conducted, resulting in significant performance improvements. The comparison of model performance before and after applying SMOTE-ENN is illustrated below:

**Model Performance Before and After SMOTE-ENN**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.67      | 0.79   | 0.73     | 765560  |
| 1     | 0.61      | 0.43   | 0.50     | 390976  |
| 2     | 0.72      | 0.70   | 0.71     | 628025  |
| **Accuracy** |           |        | 0.68     | 1784561 |

---

### III. Feature Engineering

1. **Feature Selection**:
   - Selected key features such as `source_IP`, `destination_IP`, `protocol`, `timestamp`, and `severity` for model input.
2. **Handling Categorical Data**:
   - Encoded categorical variables like IP addresses and protocols for model compatibility.
   
---

### IV. Model Training & Evaluation

- Various machine learning models were implemented and compared using the **macro-F1 score**, **precision**, **recall**, and other metrics. These models include:

  - **Random Forest**: Best-performing model with high accuracy and generalization across categories.
  - **Support Vector Machine (SVM)**: Performed well in classification but required more computational time.
  - **Logistic Regression**: Served as a baseline model, providing insight into the dataset.

#### Comparison Table:

| Model                | Accuracy | Macro-F1 Score | Precision | Recall |
|----------------------|----------|----------------|-----------|--------|
| Logistic Regression   | 0.6318   | 0.54          | 0.64      | 0.55   |
| Decision Tree         | 0.7005   | 0.67          | 0.70      | 0.66   |
| Random Forest         | 0.7011   | 0.67          | 0.70      | 0.66   |
| XGBoost               | 0.6777   | 0.62          | 0.71      | 0.61   |
| LightGBM             | 0.6756   | 0.61          | 0.72      | 0.61   |
| Gradient Boosting     | 0.6414   | 0.55          | 0.69      | 0.56   |

#### Best Model Based on Macro-F1 Score (and Accuracy in case of a tie):

- **Model**: Random Forest  
- **Accuracy**: 0.7011  
- **Macro-F1 Score**: 0.67  
- **Precision**: 0.7  
- **Recall**: 0.66  

### Evaluation Metrics:
- **Macro-F1 Score**: Evaluated the overall classification performance by balancing precision and recall across all categories.
  
---

### V. Model Deployment

The final model was deployed using **Streamlit** to provide an interactive interface where users can input incident details and receive real-time classifications. The app offers a user-friendly way for Security Operations Center (SOC) teams to classify incidents with ease.

#### Key Features of the Streamlit App:
- **Real-time Input**: Users can enter incident details directly into the interface.
- **Instant Feedback**: The app provides immediate classifications based on the input data.
- **User-Friendly Design**: The interface is designed for ease of use, making it accessible for all team members.

This deployment enhances the operational efficiency of SOC teams, enabling them to respond to cybersecurity incidents more effectively.

---

## 🧩 Model Comparison & Results

| Model              | Macro-F1 Score | Precision | Recall  |
|--------------------|----------------|-----------|---------|
| Random Forest       | 0.84           | 0.82      | 0.81    |
| SVM                | 0.81           | 0.79      | 0.77    |
| Logistic Regression | 0.76           | 0.75      | 0.74    |

---

## 🏆 Conclusion

The **Random Forest** model has demonstrated superior performance in classifying cybersecurity incidents, outpacing alternative models through its robust accuracy and adaptability. This project not only establishes a reliable framework for incident classification but also delivers a scalable solution tailored for Security Operations Center (SOC) teams.

---
## THANK YOU...
---
