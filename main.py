import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
from openai import OpenAI

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get('GROQ_API_KEY'))

url = "https://credit-fraud-ml-models.onrender.com/predict"


def prepare_input(trans_date_trans_time, cc_num, merchant, category, amt,
      first, last, gender, street, city, state, zip, lat, long,
      city_pop, job, dob, trans_num, unix_time, merch_lat,
      merch_long):

    # Initialize all category fields to 0
    category_mapping = {
        'category_food_dining': 0,
        'category_gas_transport': 0,
        'category_grocery_net': 0,
        'category_grocery_pos': 0,
        'category_health_fitness': 0,
        'category_home': 0,
        'category_kids_pets': 0,
        'category_misc_net': 0,
        'category_misc_pos': 0,
        'category_personal_care': 0,
        'category_shopping_net': 0,
        'category_shopping_pos': 0,
        'category_travel': 0
    }
    
    # Map the provided category to the correct field
    if category in category_mapping:
        category_mapping[category] = 1
    
    # Prepare the input dictionary matching the required features
    input_dict = {
        'amt': amt,
        'zip': zip,
        'lat': lat,
        'long': long,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'gender_M': 1 if gender == 'Male' else 0,
        **category_mapping  # Spread the category fields into the dictionary
    }
    
    return input_dict


def list_results(data):

    avg_prediction = int(np.mean(list(data['prediction'].values())))

    st.markdown("### Model Predictions")
    for model, pred in data['prediction'].items():
        st.write(f"- {model}: {pred}")
    st.write(f"Average Prediction: {avg_prediction}")

    avg_probability = np.mean(list(data['probability'].values()))

    st.markdown("### Model Probabilities")
    for model, prob in data['probability'].items():
        st.write(f"- {model}: {prob}")
    st.write(f"Average Probability: {avg_probability}")

    return avg_prediction, avg_probability


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, specializing in interpreting and explaining credit card fraud detection predictions. The machine learning model predicts that the transaction for customer {surname} has a {round(probability * 100, 1)}% likelihood of being fraudulent, based on the following transaction details:

Transaction Information:
{input_dict}

Top 10 Key Factors Influencing Fraud Risk:

    Feature                    | Importance
    -----------------------------------------
    Transaction Amount          | 0.608466
    Point-of-Sale Grocery       | 0.087572
    Transaction Time (Unix)     | 0.049365
    Merchant Latitude           | 0.036873
    Merchant Longitude          | 0.036654
    Customer Latitude           | 0.032725
    Customer Longitude          | 0.032214
    Customer Zip Code           | 0.030951
    Gas/Transport Category      | 0.022148
    Online Shopping             | 0.013533

You will provide a clear, concise explanation of the likelihood of this transaction being fraudulent based on the individual details and the provided key features.

- If the transaction's fraud likelihood is greater than 50%, explain in three sentences why it may be considered suspicious.
- If the transaction's fraud likelihood is less than 25%, explain in three sentences why it may not be significant.

The explanation should reference the transaction's information, relevant feature importance, and general trends from previously flagged transactions. Avoid mentioning probabilities, model predictions, or technical terms such as 'machine learning models.' Instead, directly explain the prediction in a natural, human-friendly manner. For instance, highlight unusual spending patterns, discrepancies in location, or transaction amounts compared to the customer's history.

You will keep the explanation concise and limited to three sentences.
"""

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])
    return raw_response.choices[0].message.content


st.title("Credit Card Fraud Prediction")

df = pd.read_csv("fraudTest-truncated.csv")

transaction = [
    f"{row['trans_num']} - {row['last']}" for _, row in df.iterrows()
]

selected_transaction_option = st.selectbox("Select a transaction", transaction)

if selected_transaction_option:
    selected_transaction_num = selected_transaction_option.split(" - ")[0]
    print("Selected Customer ID", selected_transaction_num)

    selected_last = selected_transaction_option.split(" - ")
    print("Surname", selected_last)

    selected_transaction = df[df["trans_num"] ==
                              selected_transaction_num].iloc[0]

    print("Selected transaction", selected_transaction)

    col1, col2 = st.columns(2)

    with col1:
        trans_date_trans_time = st.text_input(
            "Transaction Date & Time",
            selected_transaction["trans_date_trans_time"])
        cc_num = st.text_input("Credit Card Number",
                               str(selected_transaction["cc_num"]))
        merchant = st.text_input("Merchant", selected_transaction["merchant"])
        category = st.selectbox("Category", [
            "Food & Dining", "Gas Transport", "Grocery Net", "Grocery Pos",
            "Health & Fitness", "Home", "Kids & Pets", "Misc Net", "Misc Pos",
            "Personal Care", "Shopping Net", "Shopping Pos", "Travel"
        ],
                                index=["food_dining", "gas_transport", "grocery_net", "grocery_pos",
                                      "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
                                      "personal_care", "shopping_net", "shopping_pos", "travel"].index(
                                    selected_transaction["category"]))

        amt = st.number_input("Amount",
                              value=float(selected_transaction["amt"]))
        street = st.text_input("Street", selected_transaction["street"])
        city = st.text_input("City", selected_transaction["city"])
        state = st.text_input("State", selected_transaction["state"])
        zip = st.number_input("Zip Code", int(selected_transaction["zip"]))
        trans_num = st.text_input("Transaction Number",
                                  selected_transaction["trans_num"])
        unix_time = st.number_input("Unix Time",
                                    value=int(
                                        selected_transaction["unix_time"]))

    with col2:
        first = st.text_input("First Name", selected_transaction["first"])
        last = st.text_input("Last Name", selected_transaction["last"])
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_transaction["gender"] == 'M' else 1)
        lat = st.number_input("Latitude",
                              value=float(selected_transaction["lat"]))
        long = st.number_input("Longitude",
                               value=float(selected_transaction["long"]))
        city_pop = st.number_input("City Population",
                                   value=int(selected_transaction["city_pop"]))
        job = st.text_input("Job", selected_transaction["job"])
        dob = st.text_input("Date of Birth", selected_transaction["dob"])
        merch_lat = st.number_input("Merchant Latitude",
                                    value=float(
                                        selected_transaction["merch_lat"]))
        merch_long = st.number_input("Merchant Longitude",
                                     value=float(
                                         selected_transaction["merch_long"]))

    # Prepare the input dictionary for prediction
    input_dict = prepare_input(trans_date_trans_time, cc_num, merchant,
                               category, amt, first, last, gender, street,
                               city, state, zip, lat, long, city_pop, job, dob,
                               trans_num, unix_time, merch_lat, merch_long)

    # Send the request to the model
    response = requests.post(url, json=input_dict)
    if response.status_code == 200:
        result = response.json()
        avg_prediction, avg_probability = list_results(result)
        explanation = explain_prediction(avg_probability, input_dict,
                                         selected_transaction["last"])

        st.markdown("---")

        st.subheader('Explanation of Prediction')

        st.markdown(explanation)
    else:
        st.error("Error in prediction request.")
        print("Error:", response.status_code, response.text)
