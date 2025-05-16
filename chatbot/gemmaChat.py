from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()


## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANG_CHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANG_CHAIN_PROJECT")


## streamlit framework
st.title("Langchain demo With GEMMA")
input_text = st.text_input("Enter your question here:")

## Define the prompt template
prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Question {question}")
])

## Define the LLM
llm=OllamaLLM(model='gemma3')

## Define the output parser
output_parser = StrOutputParser()

chain = prompt_template|llm|output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
