import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load keys
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# single prompt
# response = llm.invoke("define langchain in one sentence")
# print(response.content)

# multiple prompts
# prompts = ["What is LangChain?", "What is Gemini?"]
# responses = llm.batch(prompts)
# for r in responses:
#     print(r.content)

# streaming response(just like typing)
# for chunk in llm.stream("Write a poem about LangChain."):
#     print(chunk.content, end="", flush=True)

# Expected output: LangChain is a framework for building LLM-powered applications by chaining together various components like models, data, and tools.
# LangChain is a framework for building applications with large language models by chaining together various components and external data sources