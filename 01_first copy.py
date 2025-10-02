import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# Load keys
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# Define template
prompt = ChatPromptTemplate.from_template(
    "State the parts of speech in this sentence: {text}"
)

# Format with actual input
final_prompt = prompt.format_messages(text="Hi!! my name is Geek.")

# Call Gemini LLM
response = llm.invoke(final_prompt)
print(response.content)
