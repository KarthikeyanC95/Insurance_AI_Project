# 🛡️ AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System

A Streamlit application that brings together six machine learning and NLP modules to support insurance risk assessment, fraud detection, customer feedback analysis, multilingual policy translation, and customer segmentation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

## 📌 Problem Statement

Insurance operations today rely heavily on manual, subjective processes: risk assessments take time and vary by assessor, fraud is difficult to catch in unstructured claims data, customer feedback goes unanalyzed at scale, multilingual policy documents are a bottleneck, and customer data is underused for segmentation and targeting.

This project's stated goal (from its own project deck) is to address four pain points:
- **Manual processes** — slow, subjective, inconsistent risk and claim evaluation.
- **Insurance fraud** — hard-to-detect patterns, often caught too late.
- **Customer satisfaction at scale** — feedback that can't be manually read across languages and volume.
- **Data underutilization** — customer data collected but not used for segmentation or prediction.

The proposed solution is a single Streamlit app with six modules, each backed by a different ML/NLP approach, covering risk scoring, fraud/anomaly detection, sentiment analysis, translation/summarization, and customer segmentation.

## 🏗️ System Modules

| # | Module | Approach |
|---|---|---|
| 1 | Home Page | Static navigation UI, no ML |
| 2 | Insurance Risk & Claim Analysis | Neural network (risk score), Logistic Regression ×2 (claim prediction, fraud claim prediction) |
| 3 | Anomaly Detection | Rule-based thresholds (claim period), Isolation Forest, Logistic Regression on engineered features |
| 4 | Customer Feedback Analysis | BERT embeddings + Random Forest |
| 5 | Policy Translation & Summarization | Pretrained Helsinki-NLP MarianMT + Facebook BART (zero-shot, not fine-tuned) |
| 6 | Customer Segmentation | K-Means clustering on PCA-reduced features |

---

## 🔍 Approach, Module by Module

### Module 2: Insurance Risk & Claim Analysis
*(Source: `Scripts/1_Insurance_Risk_and_Claim_Preprocess_model_training.ipynb`)*

**Preprocessing:** missing-value handling (mean/median/mode depending on distribution), one-hot encoding for `Policy_Type`/`Gender`, ordinal encoding for `Risk_Score`, MinMax scaling for `Annual_Income`/`Claim_Amount`/`Premium_Amount`.

**a) Risk Score Prediction (Low/Medium/High, multiclass):**
Several models were tried on this task before settling on one:
1. Random Forest (default) — overfit (100% train / 82% test), acknowledged in the notebook itself ("Model is overfitting").
2. Random Forest + 2-fold cross-validation — ~83.6% mean CV accuracy.
3. Random Forest + `GridSearchCV` hyperparameter tuning — jumps to 99% test accuracy, which is a red flag for leakage or a lucky split rather than genuine generalization, especially since the CV search itself only found 85.2% best accuracy.
4. **Neural network (Keras/TensorFlow)** — this is the model actually saved and loaded by the app (`my_model.keras`, `app.py` line 71). Train accuracy 80.5%, test accuracy 78.5%.

**b) Claim Prediction (binary — will a claim be filed):**
Plain `LogisticRegression`, saved as `logistic_regression_model.pkl`, loaded and used in the app (`app.py` line 188).

**c) Fraud Claim Prediction (binary — genuine vs. fraud):**
A second plain `LogisticRegression`, trained on features selected via a filter method + Recursive Feature Elimination (`Vehicle_Age_Property_Age`, `Premium_Amount`, `Claim_Amount`), saved as `logi_model.pkl`, loaded and used in the app (`app.py` line 232). Not to be confused with the differently-named fraud model in Module 3 below — the codebase has two separately-trained "fraud" logistic regressions.

### Module 3: Anomaly Detection
*(Sources: `Scripts/1_...ipynb` for Isolation Forest/Autoencoder exploration; `Scripts/3_fraudulent_insurance_claims_Scriptsipynb.ipynb` for the engineered-feature fraud model; `app.py` for the deployed logic)*

**a) Claim Period Analysis:** this is **not a trained model** — it's a simple rule-based threshold in `app.py` (`Days_Since_Issue < early_threshold` / `> expired_threshold`), with the thresholds set by interactive sliders. Worth knowing if you were expecting a classifier here.

**b) Anomaly Score Visualization:** Isolation Forest and an Autoencoder were both explored in Notebook 1 for anomaly scoring, compared against EllipticEnvelope and Local Outlier Factor in Notebook 3, with Isolation Forest chosen as the primary method (contamination rate manually set, ranging from 0.20–0.49 across experiments depending on notebook).

**c) Fraud Detection with engineered features:** Notebook 3 engineers two new features — `Claim_to_Income_Ratio` and `Short_Period_Claim` (claim filed within an unusually short window after policy issuance) — then trains several classifiers on a **heavily imbalanced** dataset (982 genuine claims vs. 18 fraud cases, 1.8%):

| Model | Test Accuracy | Fraud-class Precision/Recall/F1 | Note |
|---|---|---|---|
| Logistic Regression (run 1) | 98.5% | 0.80 / 0.67 / 0.73 | Reasonable |
| **Logistic Regression (run 2, saved to disk)** | 98.0% | **0.00 / 0.00 / 0.00** | **Catches zero fraud cases** despite 98% accuracy |
| Random Forest (no SMOTE) | 99.5% | 1.00 / 0.75 / 0.86 | — |
| Random Forest + SMOTE + GridSearchCV | 100% | 1.00 / 1.00 / 1.00 | Suspicious — see caveat below |
| Random Forest + SMOTE (saved to disk) | 100% | 1.00 / 1.00 / 1.00 | Suspicious — see caveat below |
| Gradient Boosting + SMOTE + GridSearchCV | 99.5% | 1.00 / 0.75 / 0.86 | — |

> ⚠️ **The model actually saved and shipped as `logistic_regression_fraud_model.pkl` is the one with zero recall on fraud.** The notebook re-runs the same `LogisticRegression` code with a stratified split partway through (to check the true class balance), and this second run — the one whose variable is still in memory when the save cell executes later — never predicts a single fraud case correctly, despite reporting 98% accuracy. This is the textbook "accuracy paradox": with 98.2% of claims genuine, predicting "genuine" every time already scores 98%. **This is the model the deployed app's "Advanced Fraud Detection" page actually loads and uses (`app.py` line 292).**

> ⚠️ **A separate, real bug in `app.py` (not just the notebook):** the deployed Isolation Forest anomaly check for a single user-submitted claim fits `IsolationForest(contamination=0.20)` on a DataFrame containing **only that one row** (`app.py`, "Updated Isolation Anomaly detection using refit approach"). Isolation Forest needs a distribution of points to isolate against — fitting and predicting on a single sample is statistically meaningless and will produce an arbitrary result every time.

> ⚠️ The 100% test accuracy for the SMOTE-tuned Random Forest isn't proven data leakage (SMOTE is applied only to the training split, after the train/test split, which is the correct order) — but the test set only contains 4 fraud cases in total, so "100%" means getting all 4 right. That's a fragile number to trust, not a validated one.

### Module 4: Customer Feedback Analysis
*(Source: `Scripts/2_Customer_Feedback_and_Sentiment_Script.ipynb`)*

**Pipeline:** review text → cleaning (special characters, stopwords, punctuation) → tokenization → lemmatization → BERT embeddings (mean-pooled) → Random Forest classifier for 3-class sentiment (Positive/Neutral/Negative).

> ⚠️ **No accuracy, precision, recall, or any other metric is ever computed for this model in the shared notebook.** A `train_test_split` is performed and a `feature_test`/`target_test` pair is created, but the notebook never calls `.predict()` on the test set or computes any score — it goes straight from `rfc.fit(feature_train, target_train)` to saving the model. The markdown header directly above the save cell literally reads "Perfect RandomForestclassifer Model" (with "Perfect" misspelled as "Prefect" in the actual saved filename, `Prefect_RandomForestclassifer_Model_for_ReviewText.pkl`) — but "Perfect" here is just a label the author chose, not a measured result. **There is no evidence in this repo of how accurate this sentiment model actually is.**

### Module 5: Policy Translation & Summarization
*(Sources: `Scripts/4_Multilingual_Insurance_Policy_Dataset_Scripts.ipynb` for the training exploration; `app.py` for what's deployed)*

The notebook explores **fine-tuning** mBART and mT5 on the project's own multilingual policy dataset. This effort ran into real problems along the way: the mBART fine-tune needed more memory than was practical ("mBART Very High Memory"), switching to mT5 produced broken output ("Still Showing `<extra_id_0>`" — a known failure mode when a T5-family model isn't properly fine-tuned, since `<extra_id_0>` is a sentinel token, not real text), and the notebook ends by going back to mBART.

**None of this fine-tuning work is what ships in the app.** `app.py` instead loads **off-the-shelf, pretrained** models directly from Hugging Face: `Helsinki-NLP/opus-mt-{lang pair}` for each translation direction (French, Spanish, Hindi, and Tamil-to-English), and `facebook/bart-large-cnn` for summarization — used zero-shot, with no fine-tuning on the project's own policy data. This is a reasonable, pragmatic choice for a demo app, but it means the translation-quality work in the notebook and the translation quality in the actual app are unrelated. **No BLEU, ROUGE, or any other translation/summarization quality metric appears anywhere in the codebase** — there's no way to say how good the translations or summaries are.

### Module 6: Customer Segmentation
*(Source: `Scripts/5_Customer_Segmentation_Scripts.ipynb`)*

**Pipeline:** feature scaling → PCA (3 components retained, ~99.4% cumulative explained variance) → clustering.

Three clustering algorithms were compared:
- **K-Means (k=4, chosen via elbow method)** — the one saved and deployed (`5_Kmeans_Unsupervised.pkl`). Silhouette score: **0.4664**.
- Agglomerative Clustering — Silhouette score: 0.4368.
- DBSCAN — failed to find meaningful clusters on this data ("Silhouette Score cannot be computed as there is only one cluster or excessive noise").

Both K-Means and Agglomerative fall in the "some overlap between clusters" band (0.3–0.5) per the notebook's own interpretation guide, rather than the "well-separated" band (>0.5) — a real but usable result, not a strong one.

> Minor note: the code comment and print statement say "k=3" while the actual `KMeans(n_clusters=4, ...)` call and the resulting `Cluster4` column confirm 4 clusters were really used (matching the "four customer segments" in the project description) — just a copy-paste label left over from an earlier version, not a functional bug.

---

## 📊 Model Performance & Metrics — What's Actually Deployed

| App page | Model file loaded | Algorithm | Real test metric (from notebook) | PPT-claimed metric |
|---|---|---|---|---|
| Risk Score Prediction | `my_model.keras` | Neural network (Keras) | **78.5% accuracy** | 92% |
| Claim Prediction | `logistic_regression_model.pkl` | Logistic Regression | **91.5% accuracy**, F1 94.6%, AUC 0.969 | 88% |
| Fraud Claim Prediction | `logi_model.pkl` | Logistic Regression | **94.0% accuracy**, F1 89.3%, AUC 0.984 | not stated separately |
| Advanced Fraud Detection (Anomaly page) | `logistic_regression_fraud_model.pkl` | Logistic Regression | **98.0% accuracy, but 0% recall on fraud** (0 of 4 fraud cases caught) | not stated |
| Customer Feedback Sentiment | `Prefect_RandomForestclassifer_Model_for_ReviewText.pkl` | Random Forest + BERT embeddings | **Not measured anywhere in the repo** | 91% |
| Policy Translation & Summarization | Pretrained Helsinki-NLP / BART (downloaded live, not a saved artifact) | MarianMT / BART | **Not measured anywhere in the repo** | not stated |
| Customer Segmentation | `5_Kmeans_Unsupervised.pkl` | K-Means (k=4) + PCA | **Silhouette score 0.4664** | not stated |

**Why this table looks different from the project's own presentation:** `Reports/insurance_ai_ppt.html` states round, presentation-friendly numbers (92% / 88% / 91% / "92% Prediction Accuracy" overall) that don't match what the notebooks' own print statements show for the exact models that get saved and loaded by `app.py`. The gap is largest for the Risk Score model (92% claimed vs. 78.5% actual) and the Advanced Fraud Detection model, where the claimed accuracy is technically true (98%) but hides that it catches no actual fraud — the more useful number here is the near-zero recall, not the accuracy. The presentation deck also cites business-impact figures ("$5M+ savings annually," "25% reduction in fraudulent claim payouts," "80% reduction in manual risk assessment time") that have no supporting cost, ROI, or business-outcome analysis anywhere in the notebooks or data — these read as illustrative/aspirational placeholders rather than measured results, and shouldn't be repeated as fact.

---

## ⚠️ Known Issues Found During Review

1. **The saved fraud-detection model in the Anomaly Detection module has 0% recall on fraud** despite 98% accuracy — a textbook accuracy-paradox result on a 98.2%/1.8% imbalanced dataset. It will not catch fraud in production as currently deployed.
2. **The deployed single-claim Isolation Forest check fits on one data point.** `IsolationForest.fit_predict()` is called on a one-row DataFrame per user submission in `app.py` — this can't meaningfully isolate anything and the anomaly flag it produces is not statistically valid.
3. **The sentiment classifier (Module 4) has no evaluation at all.** A test split is created but never scored. There is no accuracy, precision, recall, or F1 for this model anywhere in the repo.
4. **The Risk Score neural network's actual test accuracy (78.5%) is well below the 99% test accuracy obtained by a hyperparameter-tuned Random Forest in the same notebook** — but the Random Forest isn't what's deployed, and its 99% figure itself looks like it may not generalize, given the tuning search only found 85.2% during cross-validation.
5. **No translation or summarization quality metric exists** (no BLEU, ROUGE, or human evaluation) for Module 5, and the models actually shipped are pretrained/off-the-shelf, unrelated to the fine-tuning work shown in the notebook.
6. **`app.py` hardcodes Google Colab / Google Drive paths** (`/content/drive/MyDrive/Captsone project/models/...`, including a typo, "Captsone," repeated throughout) for every model file — this will not run outside of the original author's Colab environment without editing every path.
7. **The project's own presentation deck states different accuracy numbers than the notebooks that produced the deployed models**, and includes business-impact/ROI figures with no supporting analysis in the codebase — these should be treated as marketing framing, not measured results.

---

## 🛠️ Tech Stack

**ML/DL:** Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, K-Means, Isolation Forest, PCA), TensorFlow/Keras (risk score neural network), PyTorch (explored for fraud detection in Notebook 3, not deployed), imbalanced-learn (SMOTE)

**NLP:** Hugging Face Transformers — BERT (sentiment embeddings), MarianMT (translation, pretrained), BART (summarization, pretrained); mBART/mT5 (fine-tuning explored, not deployed)

**App/Data:** Streamlit, Pandas, NumPy, Plotly, Matplotlib, Joblib/Pickle

---

## 📁 Repository Structure (as found in `main`)

```
Insurance_AI_Project/
├── app.py                                     # Main Streamlit app (hardcoded Colab paths — see caveats)
├── Requirements.txt
├── README.md
│
├── Scripts/                                   # Training notebooks (source of the metrics above)
│   ├── 1_Insurance_Risk_and_Claim_Preprocess_model_training.ipynb
│   ├── 2_Customer_Feedback_and_Sentiment_Script.ipynb
│   ├── 3_fraudulent_insurance_claims_Scriptsipynb.ipynb
│   ├── 4_Multilingual_Insurance_Policy_Dataset_Scripts.ipynb
│   └── 5_Customer_Segmentation_Scripts.ipynb
│
├── Notebooks/                                  # EDA + SQL exploration notebooks per dataset
├── Models/                                     # Saved model artifacts (.pkl, .h5, .keras)
├── Data/                                       # CSV datasets (synthetic/enhanced insurance data)
├── Deployment/                                 # Deployment notes/notebook
└── Reports/                                    # Project presentation (insurance_ai_ppt.html)
```

---

## 🚀 Next Steps

- [ ] Retrain and redeploy the Advanced Fraud Detection model using class weighting or proper SMOTE-in-pipeline handling, and report precision/recall on the minority class instead of accuracy.
- [ ] Fix the single-row Isolation Forest bug — fit once on the full training distribution at startup/load time, and only call `.predict()` (not `.fit_predict()`) on new single claims.
- [ ] Add an actual test-set evaluation (accuracy, precision, recall, F1 per class) for the sentiment classifier before trusting it in production.
- [ ] Add BLEU/ROUGE evaluation for translation and summarization, or drop the fine-tuning notebook if the pretrained models are the permanent choice.
- [ ] Replace hardcoded Google Drive paths in `app.py` with relative paths or environment variables so the app runs outside Colab.
- [ ] Reconcile the presentation deck's stated accuracies with the notebooks' actual output, and remove or clearly label the unsupported ROI/business-impact figures.
- [ ] Re-run the Risk Score hyperparameter-tuned Random Forest with a fresh, larger cross-validation to check whether its 99% test score is genuine or a lucky split.

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
