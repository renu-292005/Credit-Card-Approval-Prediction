# Credit Card Approval Prediction System

An automated end-to-end machine learning system designed to analyze loan applicant financial profiles and predict credit card approval eligibility. Powered by a Flask backend and a modern, dark-themed glassmorphic Bootstrap 5 web frontend.

---

## Repository Structure

```
project/
├── dataset/
│   └── Application_Data.csv      # Applicant credit database (7.7 MB)
├── model/
│   ├── model.pkl                 # Best serialised XGBoost classifier
│   ├── scaler.pkl                # Serialised StandardScaler transformer
│   └── best_model_name.txt       # Selected model name reference
├── static/
│   ├── css/
│   │   └── style.css             # Premium custom styles and animations
│   └── js/
│       └── main.js               # Client validation and spinner loader
├── templates/
│   ├── home.html                 # Sleek dashboard landing page
│   ├── index.html                # Interactive applicant evaluation form
│   └── result.html               # Underwriting risk results & tips
├── app.py                        # Flask server backend
├── preprocessing.py              # Modular data cleaning & scaling helpers
├── train.py                      # Pipeline to train & compare classifiers
├── test_app.py                   # Automated Flask endpoint tests
├── requirements.txt              # Project packages dependency manifest
└── README.md                     # Documentation
```

---

## Data Preprocessing & Feature Engineering (`preprocessing.py`)

To ensure clean and modular code separation, the data engineering logic is isolated within `preprocessing.py`:
1. **Cleaning**: Column headers are stripped of surrounding whitespace, and features are filtered to the core underwriting variables.
2. **Imputation**: Missing values within feature lists are automatically replaced with their respective feature medians.
3. **Scaling**: Key inputs are normalized using `StandardScaler` to ensure optimal performance of distance-based estimators.
4. **Input Mapping**: Form values are converted into a pandas DataFrame dynamically with matching feature headers prior to inference to eliminate sklearn warnings.

---

## Machine Learning Pipeline (`train.py`)

Our training pipeline compared four standard classification models:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

### Target Class Balancing
The credit portfolio is heavily skewed with **25,007 Approved (Class 1)** profiles and only **121 Rejected (Class 0)** profiles. To counter this, `train.py` performs a stratified split (80/20) and **upsamples** the minority class in the training partition.

### Performance Results
Models are compared using the **Macro F1-Score** to prioritize minority class performance:

| Estimator | Accuracy | Precision (Approved) | Recall (Approved) | F1 (Approved) | Macro F1-Score |
|---|---|---|---|---|---|
| Logistic Regression | 99.20% | 100.00% | 99.20% | 99.60% | 0.7707 |
| Decision Tree | 99.28% | 99.98% | 99.30% | 99.64% | 0.7787 |
| Random Forest | 99.80% | 99.98% | 99.82% | 99.90% | 0.9102 |
| **XGBoost** | **99.96%** | **99.98%** | **99.98%** | **99.98%** | **0.9791** |

**XGBoost** was selected as the final classifier and serialized inside `model/model.pkl`.

---

## Verification & Local Setup

### 1. Requirements Installation
Ensure Python 3.12+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run Model Training
To retrain the models and save model objects to the `model/` directory:
```bash
python train.py
```

### 3. Run Automated Tests
To run route checks, boundary checks, and prediction endpoints validations:
```bash
python -m unittest test_app.py
```

### 4. Launch Local Web Server
Start the Flask development server:
```bash
python app.py
```
https://credit-card-approval-prediction-ydd5.onrender.com/

🚀 Installation Steps 

1. Clone Repository
   git clone https://github.com/renu-292005/credit-card-approval-prediction.git

2.Move into Project
cd credit-card-approval-prediction

3. Create Virtual Environment
 python -m venv venv

Windows
venv\Scripts\activate

Linux/Mac
source venv/bin/activate

4. Install Dependencies 
 pip install -r requirements.txt
💡 Usage Instructions
Train Model
python train.py
Start Flask Server
python app.py
Open
https://credit-card-approval-prediction-ydd5.onrender.com/
🤖 Machine Learning Models Used
Logistic Regression
Decision Tree
Random Forest
XGBoost
The model with the highest performance (XGBoost) is automatically
selected and saved for prediction.
📊 Performance Results
Model
Accuracy
Logistic Regression 99.20%
Decision Tree.      99.28%
Random Forest.      99.80%
XGBoost.            99.96%
🌐 Live Demo
https://credit-card-approval-prediction-ydd5.onrender.com/
📸 Screenshots
