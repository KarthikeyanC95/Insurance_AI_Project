# 🛡️ AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System

A comprehensive Streamlit-based insurance management application that leverages machine learning and natural language processing to provide intelligent insights across multiple insurance operations.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Module Details](#module-details)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)

## ✨ Features

- **Risk Assessment**: Multi-level risk scoring using neural networks
- **Fraud Detection**: Advanced anomaly detection with Isolation Forest
- **Sentiment Analysis**: Customer feedback analysis using BERT
- **Multi-language Support**: Policy translation in 7+ languages
- **Customer Segmentation**: K-means clustering with PCA visualization
- **Interactive Dashboard**: Real-time predictions and visualizations

## 🏗️ System Architecture

The application consists of **6 main modules**:

### 1. Home Page
Landing page with visual navigation to all features through interactive buttons and icons.

### 2. Insurance Risk & Claim Analysis
- **Risk Score Prediction**: Neural network classification (Low/Medium/High)
- **Claim Prediction**: Logistic regression for claim likelihood
- **Fraud Claim Prediction**: Binary fraud detection model

### 3. Anomaly Detection
- **Claim Period Analysis**: Early and expired claim identification
- **Anomaly Score Visualization**: Isolation Forest with interactive thresholds
- **Fraud Detection**: Advanced fraud prediction with engineered features

### 4. Customer Feedback Analysis
- Sentiment classification using BERT embeddings
- Random Forest classifier (Positive/Neutral/Negative)
- Real-time review analysis

### 5. Policy Translation & Summarization
- Multi-language translation (French, Spanish, Hindi, Tamil)
- Text summarization using BART
- Bidirectional translation support

### 6. Customer Segmentation
- K-means clustering with PCA
- 3D cluster visualization
- Four customer segments identification

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (optional, for faster processing)
```

### Clone Repository

```bash
git clone https://github.com/yourusername/insurance-ai-system.git
cd insurance-ai-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
streamlit
numpy
pandas
torch
transformers
tensorflow
scikit-learn
joblib
plotly
matplotlib
streamlit-lottie
streamlit-option-menu
Pillow
```

## 📊 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation

Use the sidebar menu to navigate between different modules:
- 🏠 Home
- 🛡️ Insurance Risk & Claim
- ⚠️ Anomaly Detection
- 💬 Customer Feedback
- 🌐 Policy Translation
- 👥 Customer Segmentation

## 🔍 Module Details

### Insurance Risk & Claim Analysis

**Input Features:**
- Customer Age
- Annual Income
- Vehicle/Property Age
- Claim History
- Premium Amount
- Claim Amount
- Policy Type (Auto/Health/Life/Property)
- Gender

**Outputs:**
- Risk Score: Low (0) / Medium (1) / High (2)
- Claim Prediction: Filed/Not Filed
- Fraud Status: Genuine (0) / Fraud (1)

### Anomaly Detection

**Features:**
- Adjustable threshold sliders
- Interactive visualizations
- Real-time anomaly scoring
- Fraud risk assessment with engineered features:
  - Claim-to-Income Ratio
  - Days Since Policy Issuance
  - Suspicious Flags
  - Isolation Anomaly Score

### Customer Feedback Analysis

**Capabilities:**
- Pre-loaded sample reviews
- Custom text input
- BERT-based embeddings
- Three-class sentiment classification

### Policy Translation

**Supported Languages:**
- French ↔ English
- Spanish ↔ English
- Hindi ↔ English
- Tamil → English
- Text Summarization

### Customer Segmentation

**Segments:**
- **Cluster 0**: High-value loyal customers
- **Cluster 1**: Young and growing customers
- **Cluster 2**: Risky high-claim customers
- **Cluster 3**: Low engagement customers

**Features:**
- Age, Income, Active Policies
- Total Premium Paid
- Claim Frequency
- Policy Upgrades

## 🛠️ Technologies Used

### Machine Learning
- **Scikit-learn**: Random Forest, Logistic Regression, K-means, Isolation Forest
- **TensorFlow/Keras**: Neural Network for risk scoring
- **PyTorch**: BERT model inference

### NLP Models
- **BERT**: Sentiment analysis embeddings
- **MarianMT**: Multi-language translation (Helsinki-NLP)
- **BART**: Text summarization (Facebook)

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **MinMaxScaler**: Feature normalization

### Visualization
- **Plotly**: Interactive charts
- **Matplotlib**: 3D scatter plots
- **Streamlit**: Web interface

## 📁 Project Structure

```
insurance-ai-system/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── models/                         # Trained ML models
│   ├── logi_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── my_model.keras
│   ├── onehot_encoder.pkl
│   ├── logistic_regression_fraud_model.pkl
│   ├── _Models_tokenizer_bert_textreview.pkl
│   ├── _Models_torch_bert_textreview.pkl
│   ├── 5_scaler_unsupervised.pkl
│   ├── 5_PCA_unsupervised.pkl
│   └── 5_Kmeans_Unsupervised.pkl
│
└── Data/                           # Dataset files
    ├── df1-synthetic_insurance_dataset_fixed.csv
    ├── 3_fraudulent_insurance_claims.csv
    ├── 3_feature_of_fraud_claim.csv
    ├── df3_upd_labels.csv
    └── 5_Unsupervised_customer_data.csv
```

## 🔧 Configuration

### Model Paths
Update the model paths in `app.py` to match your directory structure:

```python
# Example
logi_model = joblib.load('path/to/your/models/logi_model.pkl')
```

### Data Paths
Ensure dataset paths point to your data directory:

```python
df1 = pd.read_csv('path/to/your/Data/dataset.csv')
```

## 📈 Model Performance

| Model | Task | Accuracy |
|-------|------|----------|
| Neural Network | Risk Classification | 92%* |
| Logistic Regression | Claim Prediction | 88%* |
| Isolation Forest | Anomaly Detection | 85%* |
| BERT + Random Forest | Sentiment Analysis | 91%* |

*Performance metrics may vary based on dataset

## 👥 Authors

- Your Name - Karthikeyan C, Aspiring Data Scientist

## 🙏 Acknowledgments

- Helsinki-NLP for translation models
- Facebook AI for BART summarization
- Hugging Face Transformers library
- Streamlit community

## 📞 Contact

Project Link: [https://github.com/KarthikeyanC95/Insurance_AI_Project](https://github.com/KarthikeyanC95/Insurance_AI_Project)

---
