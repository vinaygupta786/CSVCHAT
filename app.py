from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# Load the environment variables from .env file
load_dotenv()

# Fetch API key from environment variable
api_key = os.getenv("MY_API_KEY")

# Initialize OpenAI with the API key
st.title("Welcome to CSV CHAT")
# Streamlit UI elements
csv_file = st.sidebar.file_uploader('Upload a CSV file', accept_multiple_files = True)
text = st.text_input('Enter your prompt')

# Function to generate response
def generate_response(text, csv_file):
    if csv_file is not None:
        # Read the CSV file into a DataFrame
        #df = pd.read_csv(csv_file)
        df = [pd.read_csv(file) for file in csv_file]
        combined_df = pd.concat(df, ignore_index=True)
        
        # Initialize the OpenAI LLM
        llm = OpenAI(api_token=api_key, model="gpt-4o")

        # Create the SmartDataframe
        smart_df = SmartDataframe(combined_df, config={"llm": llm})

        # Generate a response based on the user input
        response = smart_df.chat(text)
        return response

    else:
        st.error("Please upload a CSV file.")

# Display the response if both inputs are provided
if csv_file and text:
    response = generate_response(text, csv_file)
    st.write(response)
