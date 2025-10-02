import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load API key from environment
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7, 
    api_key=google_api_key
)

# --- Chain 1: Write a Synopsis ---
template1 = """You are a playwright. Given the title of a play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}

Playwright: This is a synopsis for the above play:"""

prompt_template1 = PromptTemplate(
    input_variables=["title", "era"], 
    template=template1
)

synopsis_chain = LLMChain(
    llm=llm, 
    prompt=prompt_template1, 
    output_key="synopsis"
)

# --- Chain 2: Write a Review ---
template2 = """You are a play critic from the New York Times. Given the synopsis of a play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}

Review from a New York Times play critic of the above play:"""

prompt_template2 = PromptTemplate(
    input_variables=["synopsis"], 
    template=template2
)

review_chain = LLMChain(
    llm=llm, 
    prompt=prompt_template2, 
    output_key="review"
)

# --- Sequential Chain ---
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["title", "era"],
    output_variables=["synopsis", "review"],
    verbose=True
)

# Run example
result = overall_chain({"era": "Victorian London", "title": "The Clockwork Heart"})
print(result)

""" 
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=google_api_key
)

synopsis_prompt = PromptTemplate.from_template(
    """"""You are a playwright. Given the title of a play and the era it is set in,
it is your job to write a synopsis for that title.

Title: {title}
Era: {era}

Playwright: This is a synopsis for the above play:""""""
)

review_prompt = PromptTemplate.from_template(
    """"""You are a play critic from the New York Times.
Given the synopsis of a play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}

Review from a New York Times play critic of the above play:""""""
)

synopsis_chain = synopsis_prompt | llm
review_chain = review_prompt | llm
overall_chain = synopsis_chain | review_chain

result = overall_chain.invoke({"title": "The Clockwork Heart", "era": "Victorian London"})
print(result)
"""