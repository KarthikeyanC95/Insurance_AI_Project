import os
import pickle
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import torch
from streamlit_lottie import st_lottie
from PIL import Image
from transformers import BertTokenizer, BertModel, MarianMTModel, MarianTokenizer, BartForConditionalGeneration, BartTokenizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import date
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.ensemble import IsolationForest
from mpl_toolkits.mplot3d import Axes3D
import streamlit.components.v1 as components
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# Main app navigation
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Insurance Risk & Claim", "Anomaly Detection",
         "Customer Feedback", "Policy Translation", "Customer Segmentation"],
        icons=['house', 'shield', 'exclamation-triangle',
               'chat', 'translate', 'people'],
        menu_icon="cast",
        default_index=0
    )

# Home Page
if selected == "Home":
    st.title(":red[AI-Powered Intelligent Insurance Risk Assessment and customer Insights System]")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Insurance Risk & Claim"):
            selected = "Insurance Risk & Claim"
        st.image("https://cdn-icons-png.flaticon.com/512/8438/8438971.png", use_container_width=True)

        if st.button("Anomaly Detection"):
            selected = "Anomaly Detection"
        st.image("https://cdn-icons-png.flaticon.com/512/11331/11331293.png", use_container_width=True)

    with col2:
        if st.button("Customer Feedback"):
            selected = "Customer Feedback"
        st.image("https://static.vecteezy.com/system/resources/previews/041/317/536/original/3d-feedback-icon-on-transparent-background-png.png", use_container_width=True)

        if st.button("Policy Translation"):
            selected = "Policy Translation"
        st.image("https://icon-library.com/images/translate-icon/translate-icon-4.jpg", use_container_width=True)

    with col3:
        if st.button("Customer Segmentation"):
            selected = "Customer Segmentation"
        st.image("https://cdn-icons-png.flaticon.com/512/2761/2761493.png", use_container_width=True)

# 1. Insurance Risk & Claim Page
elif selected == "Insurance Risk & Claim":
    # Read dataset for scaling
    # Load models and encoders
    logi_model = joblib.load('/content/drive/MyDrive/Captsone project/models/logi_model.pkl')
    logis_model = joblib.load('/content/drive/MyDrive/Captsone project/models/logistic_regression_model.pkl')
    loaded_model = load_model('/content/drive/MyDrive/Captsone project/models/my_model.keras')
    with open('/content/drive/MyDrive/Captsone project/models/onehot_encoder.pkl', 'rb') as file:
        onehot_encoder = pickle.load(file)

    # Read dataset for scaling
    df1 = pd.read_csv('/content/drive/MyDrive/Captsone project/Data/df1-synthetic_insurance_dataset_fixed.csv')

    # Get min and max values for scaling
    min_values = df1[['Annual_Income', 'Premium_Amount', 'Claim_Amount']].min()
    max_values = df1[['Annual_Income', 'Premium_Amount', 'Claim_Amount']].max()

    # Fit the scaler
    min_max_values = np.array([min_values, max_values])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(min_max_values)

    st.title("Insurance Risk & Claim Analysis")
    tab1, tab2, tab3 = st.tabs(["Risk Score Prediction", "Claim Prediction", "Fraud Claim Prediction"])

    with tab1:
        st.header("Risk Score Prediction")
        # # Load models
        # Risk_model = joblib.load(r'/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/Risk_model.pkl')
        # scaler =  joblib.load(r'/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/scaler_Risk.pkl')
        # with open('/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/onehot_Risk_encoder.pkl', 'rb') as file:
        #     onehot_encoder = pickle.load(file)

        # User inputs
        col1, col2,col3, col4 = st.columns(4)
        with col1:
            customer_age = st.number_input('Customer Age', min_value=18, max_value=80, step=1, key='age_risk')
            annual_income_raw = st.number_input('Annual Income', key='income_risk')
            vehicle_age_property_age = st.number_input('Vehicle/Property Age', min_value=0, max_value=30, step=1, key='vehicle_age_risk')


        with col2:
            claim_history = st.number_input('Claim History', min_value=0, max_value=5, step=1, key='claim_history_risk')
            premium_amount_raw = st.number_input('Premium Amount', key='premium_risk')


        with col3:
            claim_amount_raw = st.number_input('Claim Amount', key='claim_amt_risk')
            fraudulent_claim = st.number_input('Fraudulent Claim', key='fraud_risk')

        with col4:
            policy_type = st.selectbox('Policy Type', ['Auto', 'Health', 'Life', 'Property'], key='policy_type_risk')
            gender = st.selectbox('Gender', ['Female', 'Male', 'Other'], key='gender_risk')

        categorical_data = pd.DataFrame({'Policy_Type': [policy_type], 'Gender': [gender]})
        encoded_data = onehot_encoder.transform(categorical_data).reshape(1, -1)

        scaled_features = scaler.transform([[annual_income_raw, premium_amount_raw, claim_amount_raw]])
        annual_income, premium_amount, claim_amount = scaled_features[0]

        new_data = np.array([[customer_age, annual_income, vehicle_age_property_age, claim_history, premium_amount, claim_amount, fraudulent_claim]])
        final_input = np.concatenate([new_data, encoded_data], axis=1)

        if st.button('Predict Advanced Fraud Risk'):
            try:
                predictions = loaded_model.predict(final_input)
                predicted_class = np.argmax(predictions, axis=1)
                risk_labels = {0: 'Low (0)', 1: 'Medium (1)', 2: 'High (2)'}
                risk_score = risk_labels.get(predicted_class[0], 'Unknown')
                st.write(f"Predicted class: {predicted_class[0]}")
                st.write(f"Risk Score: {risk_score}")
                st.write(f"Class probabilities: {predictions}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with tab2:
        st.header("Claim Prediction")
        # Claim_Amount_model = joblib.load(r'/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/Claim_Amount_model.pkl')
        # with open('/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/onehot_Claim_encoder.pkl', 'rb') as file:
        #     onehot_encoder = pickle.load(file)
        # with open('/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/lbl_Claim_encoder.pkl', 'rb') as file:
        #     label_encoder = pickle.load(file)

        # User inputs
        col1, col2,col3 = st.columns(3)
        with col1:
            customer_age = st.number_input('Customer Age', min_value=18, max_value=80, step=1, key='age_claim')
            annual_income_raw = st.number_input('Annual Income', key='income_claim')
            Vehicle_Age_Property_Age = st.number_input('Vehicle/Property Age', min_value=0, max_value=30, step=1, key='vehicle_age_claim')


        with col2:
            claim_history = st.number_input('Claim History', min_value=0, max_value=5, step=1, key='claim_history_claim')
            premium_amount_raw = st.number_input('Premium Amount', key='premium_claim')

        with col3:
            policy_type = st.selectbox('Policy Type', ['Auto', 'Health', 'Life', 'Property'], key='policy_type_claim')
            gender = st.selectbox('Gender', ['Female', 'Male', 'Other'], key='gender_claim')


        policy_type_encoded = [1 if policy_type == p else 0 for p in ['Auto', 'Health', 'Life', 'Property']]
        gender_encoded = [1 if gender == g else 0 for g in ['Female', 'Male', 'Other']]

        scaled_features = scaler.transform([[annual_income_raw, premium_amount_raw, 0]])  # Add a placeholder for Claim_Amount
        annual_income, premium_amount, _ = scaled_features[0]  # Ignore the third value

        new_data = pd.DataFrame({
            'Customer_Age': [customer_age],
            'Policy_Type_Auto': [policy_type_encoded[0]],
            'Policy_Type_Health': [policy_type_encoded[1]],
            'Policy_Type_Life': [policy_type_encoded[2]],
            'Policy_Type_Property': [policy_type_encoded[3]],
            'Gender_Female': [gender_encoded[0]],
            'Gender_Male': [gender_encoded[1]],
            'Gender_Other': [gender_encoded[2]],
            'Annual_Income': [annual_income],
            'Vehicle_Age_Property_Age': [vehicle_age_property_age],
            'Premium_Amount': [premium_amount],
            'Claim_History': [claim_history]
        })

        if st.button('Predicts Claim Prediction'):
            try:
                predictions = logis_model.predict_proba(new_data)
                predicted_class = logis_model.predict(new_data)
                Claim_labels = {0: 'Filed Claim 2 or Below / did not file a claim Single time (0)', 1: 'Already Filed Claims More than 2 (1)'}
                Claim_score = Claim_labels.get(predicted_class[0], 'Unknown')
                st.write(f"Predicted class: {predicted_class[0]}")
                st.write(f"Claim Score: {Claim_score}")
                st.write(f"Class probabilities: {predictions}")
            except Exception as e:
                st.error(f"An error occurred: {e}")


    with tab3:
        st.header('Fraud Claim Prediction')
        # Fraud_Amount_model = joblib.load(r'/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/Fraud_claim_model.pkl')
        # # scaler_Fraud =  joblib.load('E:\Captsone project\Insurance_Risk_Claim_Dataset\scaler_Risk.pkl')
        # with open('/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/scaler_Risk.pkl', 'rb') as file:
        #     scaler_Fraud = pickle.load(file)
        # with open('/content/drive/MyDrive/Captsone project/Insurance_Risk_Claim_Dataset/lbl_Claim_encoder.pkl', 'rb') as file:
        #     label_encoder = pickle.load(file)

        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            vehicle_age_property_age = st.number_input('Vehicle_Age_Property_Age', min_value=0, max_value=30, step=1, key='vehicle_age_Fraud')
            claim_amount_raw = st.number_input('Claim_Amount', key='claim_amt_Fraud')

        with col2:
            premium_amount_raw = st.number_input('Premium_Amount', key='premium_Fraud')
            claim_history = st.number_input('Claim_History', min_value=0, max_value=5, step=1, key='claim_history_Fraud')


        scaled_features = scaler.transform([[0, premium_amount_raw, claim_amount_raw]])  # Placeholder for missing feature
        _, premium_amount, claim_amount = scaled_features[0]  # Ignore the first value


        new_data = pd.DataFrame({
            'Vehicle_Age_Property_Age': [vehicle_age_property_age],
            'Premium_Amount': [premium_amount],
            'Claim_Amount': [claim_amount],
            'Claim_History': [claim_history]
        })

        if st.button('Predict Simple Fraud'):
            try:
                predictions = logi_model.predict_proba(new_data)
                predicted_class = logi_model.predict(new_data)
                risk_labels = {0: 'Genuine (0)', 1: 'Fraud (1)'}
                risk_score = risk_labels.get(predicted_class[0], 'Unknown')
                st.write(f"Predicted class: {predicted_class[0]}")
                st.write(f"Risk Score: {risk_score}")
                st.write(f"Class probabilities: {predictions}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# 2. Anomaly Detection Page
elif selected == "Anomaly Detection":
    st.title("Anomaly Detection in Insurance Claims")

    tab1, tab2, tab3 = st.tabs(["Claim Period Analysis", "Anomaly Score with graph", "Fraud Detection"])

    with tab1:
        st.header("Claim Period Analysis")
        df = pd.read_csv(r'/content/drive/MyDrive/Captsone project/Data/3_fraudulent_insurance_claims.csv')

        early_threshold = st.slider("Early Claim Threshold (days)", 30, 180, 90)
        expired_threshold = st.slider("Expired Claim Threshold (days)", 365, 1750, 365)

        df["Early_Claim_Flag"] = df["Days_Since_Issue"] < early_threshold
        df["Expired_Claim_Flag"] = df["Days_Since_Issue"] > expired_threshold

        early_claims = df[df["Early_Claim_Flag"]]
        expired_claims = df[df["Expired_Claim_Flag"]]

        st.write(f"Early Claims: {len(early_claims)}")
        st.write(f"Expired Claims: {len(expired_claims)}")

        fig = px.pie(names=["Early", "Expired", "Normal"],
                    values=[len(early_claims), len(expired_claims), len(df) - len(early_claims) - len(expired_claims)])
        st.plotly_chart(fig)

    with tab2:
        st.header("Anomaly Score with graph")
        data = pd.read_csv(r'/content/drive/MyDrive/Captsone project/Data/3_feature_of_fraud_claim.csv').drop(columns=['Unnamed: 0'])
        concated_df = pd.read_csv(r'/content/drive/MyDrive/Captsone project/Data/3_fraudulent_insurance_claims.csv')

        anomaly_percent = st.slider("Anomaly Percentile", 1, 50, 33)

        iso_forest = IsolationForest(n_estimators=200, random_state=42)
        iso_forest.fit(data)

        concated_df["Anomaly_Score"] = iso_forest.decision_function(data)
        threshold = np.percentile(concated_df["Anomaly_Score"], anomaly_percent)
        concated_df["Anomaly_Label"] = (concated_df["Anomaly_Score"] < threshold).astype(int)

        anomalies = concated_df[concated_df["Anomaly_Label"] == 1]
        st.write(f"Detected {len(anomalies)} anomalies")

        fig = px.histogram(concated_df, x="Anomaly_Score")
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig)

    with tab3:
        st.header("Fraud Detection")
        # Load the trained model
        log_reg = joblib.load('/content/drive/MyDrive/Captsone project/models/logistic_regression_fraud_model.pkl')

        # Read the dataset
        df1 = pd.read_csv('/content/drive/MyDrive/Captsone project/Data/df3_upd_labels.csv')

        # Get min and max values for the required columns
        min_values = df1[['Annual_Income', 'Claim_Amount']].min()
        max_values = df1[['Annual_Income', 'Claim_Amount']].max()
        min_max_values = np.array([min_values, max_values])

        # Fit the scaler on known ranges
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(min_max_values)

        # Feature list used during training
        training_features = ['Claim_Amount', 'Suspicious_Flags', 'Claim_Type_Home_Damage', 'Claim_Type_Medical',
                             'Claim_Type_Vehicle', 'Claim_Year', 'Claim_Month', 'Claim_Day', 'Annual_Income',
                             'Claim_to_Income_Ratio', 'Days_Since_Issuance', 'Short_Period_Claim', 'Isolation_Anomaly',
                             'policy_issue_Year', 'policy_issue_Month', 'policy_issue_Day']

        # Function to calculate days since policy issuance
        def calculate_days_since_issuance(claim_date, policy_issue_date):
            delta = claim_date - policy_issue_date
            return delta.days

        # Function to prepare input data
        def prepare_input_data(user_input):
            # Scale Claim_Amount and Annual_Income
            scaled_values = scaler.transform([[user_input['Annual_Income'], user_input['Claim_Amount']]])
            user_input['Annual_Income'], user_input['Claim_Amount'] = scaled_values[0]

            # Map Claim_Type to one-hot encoded columns
            claim_type_mapping = {
                'Home Damage': [1, 0, 0],
                'Medical': [0, 1, 0],
                'Vehicle': [0, 0, 1]
            }
            user_input['Claim_Type_Home_Damage'], user_input['Claim_Type_Medical'], user_input['Claim_Type_Vehicle'] = claim_type_mapping[user_input['Claim_Type']]

            # Calculate engineered features
            user_input['Claim_to_Income_Ratio'] = user_input['Claim_Amount'] / user_input['Annual_Income']
            user_input['Suspicious_Flags'] = 1 if user_input['Claim_to_Income_Ratio'] > 0.5 else 0

            # Calculate days since policy issuance
            claim_date = date(user_input['Claim_Year'], user_input['Claim_Month'], user_input['Claim_Day'])
            policy_issue_date = date(user_input['policy_issue_Year'], user_input['policy_issue_Month'], user_input['policy_issue_Day'])
            user_input['Days_Since_Issuance'] = calculate_days_since_issuance(claim_date, policy_issue_date)

            # Determine if it's a short-period claim
            user_input['Short_Period_Claim'] = 1 if user_input['Days_Since_Issuance'] < 365 else 0

            # Updated Isolation Anomaly detection using refit approach
            sample_data = pd.DataFrame([[user_input['Claim_Amount'], user_input['Claim_Year'], user_input['Claim_Month'], user_input['Claim_Day'], user_input['Claim_to_Income_Ratio'], user_input['Days_Since_Issuance']]])
            iso_forest = IsolationForest(contamination=0.20, random_state=42)
            user_input['Isolation_Anomaly'] = iso_forest.fit_predict(sample_data)[0]

            # Align feature names and order
            input_data = pd.DataFrame([user_input])[training_features]

            return input_data, user_input, claim_date, policy_issue_date

        # Collect user input
        claim_amount = st.number_input('Claim Amount', min_value=0.00)
        claim_type = st.selectbox('Claim Type', ['Home Damage', 'Medical', 'Vehicle'])
        claim_year = st.number_input('Claim Year', min_value=2000, max_value=2100)
        claim_month = st.number_input('Claim Month', min_value=1, max_value=12)
        claim_day = st.number_input('Claim Day', min_value=1, max_value=31)
        annual_income = st.number_input('Annual Income', min_value=0.00)
        policy_issue_year = st.number_input('Policy Issue Year', min_value=2000, max_value=2100)
        policy_issue_month = st.number_input('Policy Issue Month', min_value=1, max_value=12)
        policy_issue_day = st.number_input('Policy Issue Day', min_value=1, max_value=31)

        # Prepare input and predict if the user clicks the button
        if st.button('Predict Fraud Risk'):
            user_input = {
                'Claim_Amount': claim_amount,
                'Claim_Type': claim_type,
                'Claim_Year': claim_year,
                'Claim_Month': claim_month,
                'Claim_Day': claim_day,
                'Annual_Income': annual_income,
                'policy_issue_Year': policy_issue_year,
                'policy_issue_Month': policy_issue_month,
                'policy_issue_Day': policy_issue_day
            }

            prepared_data, user_input_features, claim_date, policy_issue_date = prepare_input_data(user_input)
            prediction = log_reg.predict(prepared_data)
            result = 'FRAUD' if prediction[0] == 1 else 'GENUINE'
            st.success(f'Predicted class: {result}')

            st.write(f"Claim Date: {claim_date}")
            st.write(f"Policy Issue Date: {policy_issue_date}")
            st.write(f"Days Since Issuance: {user_input_features['Days_Since_Issuance']}")

            # Show calculated feature values
            st.write(f"Claim to Income Ratio: {user_input_features['Claim_to_Income_Ratio']}")
            st.write(f"Suspicious Flags: {'True' if user_input_features['Suspicious_Flags'] == 1 else 'False'}")
            st.write(f"Short Period Claim: {'True' if user_input_features['Short_Period_Claim'] == 1 else 'False'}")
            st.write(f"Isolation Anomaly: {'Anomaly' if user_input_features['Isolation_Anomaly'] == -1 else 'Normal'}")
            st.write(f"Scaled Annual Income: {user_input_features['Annual_Income']}")
            st.write(f"Scaled Claim Amount: {user_input_features['Claim_Amount']}")



# 3. Customer Feedback Analysis
elif selected == "Customer Feedback":
    st.title("Customer Feedback Analysis")

    # Load models
    tokenizer = joblib.load(r'/content/drive/MyDrive/Captsone project/models/_Models_tokenizer_bert_textreview.pkl')
    model = joblib.load(r'/content/drive/MyDrive/Captsone project/models/_Models_torch_bert_textreview.pkl')
    rfc_model = joblib.load(r'/content/drive/MyDrive/Captsone project/models/_Models_Prefect_RandomForestclassifer_Model_for_ReviewText.pkl')

    # Sample reviews
    reviews = {
        "Positive": "Excellent service, very helpful staff, quick response times",
        "Neutral": "Itself turn law purpose budget require course",
        "Negative": "Terrible service, slow response, unhelpful staff"
    }

    review_text = st.selectbox("Select a sample review or enter your own:",
                             list(reviews.values()) + ["Custom"])
    if review_text == "Custom":
        review_text = st.text_area("Enter your review text:")

    if st.button("Analyze Sentiment"):
        inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            output_attentions=True
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

        prediction = rfc_model.predict([embedding])[0]
        st.write("rfc_model:", prediction)
        if prediction == "Positive":
            st.success("Positive Sentiment")
        elif prediction == "Neutral":
            st.warning("Neutral Sentiment")
        else:
            st.error("Negative Sentiment")




# 4. Policy Translation Page
elif selected == "Policy Translation":
    st.title("Policy Translation & Summarization")

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Frenchh to English Models
    fr_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    fr_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en", device_map='auto')
    #English to French Models
    en_fr_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    en_fr_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr", device_map='auto')
    #Tamil to English Models
    ta_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    ta_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en", device_map='auto')
    #English to Hindi Models
    en_hi_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    en_hi_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi", device_map='auto')
    #Hindi to English Models
    hi_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en", device_map='auto')
    #Spanish to English Model
    es_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    es_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en").to(device)
    #English to Spanish Model
    en_es_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    en_es_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)
    #summarize Models
    summarizer_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", device_map='auto')

    text = st.text_area("Enter text to translate or summarize:")
    operation = st.radio("Select operation:",
                        ["Translate French to English",
                         "Translate English to French",
                         "Translate Spanish to English",
                         "Translate English to Spanish",
                         "Translate Tamil to English",
                         "Translate English to Hindi",
                         "Translate Hindi to English",
                         "Summarize Text"])

    if st.button("Process"):
      if operation == "Translate French to English":
        inputs = fr_en_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = fr_en_model.generate(inputs)
        translated = fr_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
      elif operation == "Translate English to French":
        inputs = en_fr_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = en_fr_model.generate(inputs)
        translated = en_fr_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
      elif operation == "Translate Spanish to English":
        inputs = es_en_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = es_en_model.generate(inputs)
        translated = es_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
      elif operation == "Translate English to Spanish":
        inputs = en_es_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = en_es_model.generate(inputs)
        translated = en_es_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
      elif operation == "Translate Tamil to English":
        inputs = ta_en_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = ta_en_model.generate(inputs)
        translated = ta_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
      elif operation == "Translate English to Hindi":
        inputs = en_hi_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = en_hi_model.generate(inputs)
        translated = en_hi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
    elif operation == "Translate Hindi to English":
        inputs = hi_en_tokenizer.encode(text, return_tensors="pt", truncation=True).to(device)
        outputs = hi_en_model.generate(inputs)
        translated = hi_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(f"Translated: {translated}")
    else:
        inputs = summarizer_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=50).to(device)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success(f"Summary: {summary}")

# 5. Customer Segmentation Page
elif selected == "Customer Segmentation":
    st.title("Customer Segmentation")

    # Load models
    scaler = joblib.load(r"/content/drive/MyDrive/Captsone project/models/5_scaler_unsupervised.pkl")
    pca = joblib.load(r"/content/drive/MyDrive/Captsone project/models/5_PCA_unsupervised.pkl")
    kmeans = joblib.load(r"/content/drive/MyDrive/Captsone project/models/5_Kmeans_Unsupervised.pkl")

    tab1, tab2 = st.tabs(["Cluster Visualization", "Predict Cluster"])

    with tab1:
        df = pd.read_csv(r"/content/drive/MyDrive/Captsone project/Data/5_Unsupervised_customer_data.csv")
        df_pca = pca.transform(df.iloc[:, 1:])
        df["Cluster"] = kmeans.labels_

        fig = px.pie(df, names="Cluster", title="Cluster Distribution")
        st.plotly_chart(fig)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=df["Cluster"])
        st.pyplot(fig)

    with tab2:
        age = st.number_input("Age", min_value=18, max_value=100)
        income = st.number_input("Income", min_value=10000)
        Number_of_Active_Policies = st.number_input("Number of Active Policies", min_value=1)
        total_premium_paid = st.number_input("Total Premium Paid", min_value=500)
        claim_frequency = st.number_input("Claim Frequency", min_value=0)
        policy_upgrades = st.number_input("Policy Upgrades", min_value=0)

        if st.button("Predict Cluster"):
            new_data = pd.DataFrame([[age, income, Number_of_Active_Policies, total_premium_paid, claim_frequency, policy_upgrades]],
                                  columns=["Age", "Income", "Number_of_Active_Policies", "Total_Premium_Paid", "Claim_Frequency", "Policy_Upgrades"])

            new_data[["Income", "Total_Premium_Paid"]] = scaler.transform(new_data[["Income", "Total_Premium_Paid"]])
            new_data_pca = pca.transform(new_data)
            cluster = kmeans.predict(new_data_pca)[0]

            st.success(f"Predicted Cluster: {cluster}")

            cluster_info = {
                0: "High-value loyal customers",
                1: "Young and growing customers",
                2: "Risky high-claim customers",
                3: "Low engagement customers"
            }

            st.write(f"Cluster Characteristics: {cluster_info.get(cluster, 'Unknown')}")
