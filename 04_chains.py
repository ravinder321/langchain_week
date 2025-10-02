import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

cities = ["New York City", "Los Angeles", "Chicago", "Houston"]

# Prompt template
prompt = PromptTemplate.from_template("What is the capital of {place}?")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Runnable chain (new style)
for city in cities:
    output = (prompt | llm).invoke({"place": city})
    print(output)
