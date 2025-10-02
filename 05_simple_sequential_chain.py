import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Prompt 1: Capital
prompt1 = PromptTemplate.from_template("What is the capital of {place}?")
llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=google_api_key)
chain1 = LLMChain(llm=llm1, prompt=prompt1)

# Prompt 2: Currency
prompt2 = PromptTemplate.from_template("What is the currency of {place}?")
llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=google_api_key)
chain2 = LLMChain(llm=llm2, prompt=prompt2)

# Sequential chain
chain = SimpleSequentialChain(
    chains=[chain1, chain2],
)

# Run the chain
output = chain.run("Punjab")
print(output)

# Modern approach 
"""
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Define prompts
prompt1 = PromptTemplate.from_template("What is the capital of {place}?")
prompt2 = PromptTemplate.from_template("What is the currency of {place}?")

# Initialize LLMs
llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=google_api_key)
llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=google_api_key)

# Create RunnableSequence
chain = (prompt1 | llm1 | prompt2 | llm2)

# Invoke with dict input
output = chain.invoke({"place": "Punjab"})
print(output)

"""