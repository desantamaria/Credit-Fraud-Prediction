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

    input_dict = {
        'trans_date_trans_time': trans_date_trans_time,
        'cc_num': cc_num,
        'merchant': merchant,
        'amt': amt,
        'first': first,
        'last': last,
        'gender_M': 1 if gender == 'Male' else 0,
        'street': street,
        'city': city,
        'state': state,
        'zip': zip,
        'lat': lat,
        'long': long,
        'city_pop': city_pop,
        'job': job,
        'dob': dob,
        'trans_num': trans_num,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
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
    prompt = f"""You are an expert data scientist at a bank, specializing in interpreting and explaining customer churn predictions. The machine learning model predicts that the customer, {surname}, has a {round(probability * 100, 1)}% chance of churning, based on the following details:

Customer Information:
{input_dict}

You will carefully consider that the age in the given customer info falls into one of these categories:
    Young: Age < 30
    Middle-Aged: 30 < Age < 45
    Senior: 45 < Age < 60
    Elderly: 60 < Age < 100

For example, if the customer is 40 years old, you should explain that they are middle-aged. Another example, if the customer is 56 years old, you should explain that they are senior.

Top 10 Key Factors Influencing Churn Risk:

    Feature                | Importance
    -------------------------------------
    AgeGroup_Senior         | 0.359508
    NumOfProducts           | 0.112505
    IsActiveMember          | 0.078493
    Age                     | 0.051761
    Geography_Germany       | 0.043180
    Gender_Male             | 0.022967
    Balance                 | 0.021839
    CLV                     | 0.020594
    Geography_Spain         | 0.017027
    EstimatedSalary         | 0.013020

Below are summary statistics for customers who churned:
{df[df['Exited'] == 1].describe()}

You will provide a clear, concise explanation of the customer's likelihood of churning based on their individual details and the provided key features. 

- If the customer's risk of churning is greater than 40%, explain in three sentences why they may be at risk of churning.

- If the customer's risk of churning is less than 25%, explain in three sentences why they may not be at significant risk.

The explanation should reference the customer's information, relevant feature importance, and general trends from churned customers. Avoid mentioning the churn probability, model predictions, or technical terms such as 'machine learning models' or 'top 10 features.' Instead, directly explain the prediction in a natural, human-friendly manner. Do not mention the features of the model by name, for example, "The age falls into the 'Middle-Age' category 30<Age<45.

 You will keep the explanation concise and limied to three sentences.
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
                                index=["personal_care", "gas_transport"].index(
                                    selected_transaction["category"]))

        amt = st.number_input("Amount",
                              value=float(selected_transaction["amt"]))
        street = st.text_input("Street", selected_transaction["street"])
        city = st.text_input("City", selected_transaction["city"])
        state = st.text_input("State", selected_transaction["state"])
        zip = st.text_input("Zip Code", str(selected_transaction["zip"]))
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
        list_results(result)
    else:
        st.error("Error in prediction request.")

    #     age = st.number_input("Age",
    #                           min_value=18,
    #                           max_value=100,
    #                           value=int(selected_customer["Age"]))

    #     tenure = st.number_input("Tenure",
    #                              min_value=0,
    #                              max_value=50,
    #                              value=int(selected_customer["Tenure"]))

    # with col2:
    #     balance = st.number_input("Balance",
    #                               min_value=0.0,
    #                               value=float(selected_customer["Balance"]))

    #     num_products = st.number_input("Number of Products",
    #                                    min_value=1,
    #                                    max_value=10,
    #                                    value=int(
    #                                        selected_customer["NumOfProducts"]))

    #     has_credit_card = st.checkbox("Has Credit Card",
    #                                   value=bool(
    #                                       selected_customer["HasCrCard"]))

    #     is_active_member = st.checkbox(
    #         "Is Active Member",
    #         value=bool(selected_customer["IsActiveMember"]))

    #     estimated_salary = st.number_input(
    #         "Estimated Salary",
    #         min_value=0.0,
    #         value=float(selected_customer["EstimatedSalary"]))

    # input_dict = prepare_input(credit_score, location, gender, age, tenure,
    #                            balance, num_products, has_credit_card,
    #                            is_active_member, estimated_salary)

    # response = requests.post(url, json=input_dict)

    # if response.status_code == 200:
    #     result = response.json()

    #     avg_prediction, avg_probability = list_results(result)

    #     explanation = explain_prediction(avg_probability, input_dict,
    #                                      selected_customer["Surname"])

    #     st.markdown("---")

    #     st.subheader('Explanation of Prediction')

    #     st.markdown(explanation)

    #     email = generate_email(avg_probability, input_dict, explanation,
    #                            selected_customer["Surname"])

    #     st.markdown("---")

    #     st.subheader("Personalized Email")

    #     st.markdown(email)

    # else:
    #     print("Error:", response.status_code, response.text)
