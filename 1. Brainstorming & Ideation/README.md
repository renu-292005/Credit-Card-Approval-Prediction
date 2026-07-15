## 1. Brainstorming & Ideation

### 1.1. Introduction & Context
In the modern financial sector, credit cards have evolved from premium financial instruments into essential commodities for consumer transactions and credit building. For banking institutions, credit evaluation (underwriting) is a foundational process that directly impacts profitability and solvency. Traditionally, credit card approval decisions relied heavily on manual verification, credit officers' subjective judgment, and rule-based scorecards (e.g., FICO cut-offs). While functional, manual processes are inherently slow, error-prone, inconsistent, and fail to capture non-linear interactions between complex financial indicators.

### 1.2. Problem Statement
The primary challenge faced by credit issuers is the **Credit Risk Classification Problem**: predicting whether a credit card applicant will default on their financial obligations or remain a "good client" based on their demographic variables, employment attributes, and historical debt portfolio. Machine learning models must process extremely imbalanced datasets (e.g., where approvals vastly outnumber rejections) and accurately flag high-risk individuals without introducing discriminatory biases or reducing overall approval throughput.

### 1.3. Objectives
The objectives of the proposed system are:
1. **Automation**: To eliminate manual processing delays by rendering instant credit card eligibility predictions (sub-50ms response times).
2. **Predictive Precision**: To implement a machine learning classifier that compares standard statistical models (Logistic Regression, Decision Trees, Random Forests, XGBoost) and selects the model maximizing minority-class recall and macro F1-score.
3. **Class Balancing Optimization**: To address extreme target class imbalances (e.g., where approved accounts exceed rejected accounts by a ratio of 200:1) to prevent the classifier from developing approval bias.
4. **Underwriting Dashboard**: To connect the ML pipeline with a secure Flask web service and a responsive Bootstrap 5 frontend for risk analysts and underwriting agents.

---