import langchain

import os
os.environ["OPENAI_API_KEY"]= "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"

#Proprietary LLM from eg.openai
#pip install openai
from langchain.llms import OpenAI
llm = OpenAI(model_name = "text-davinci-003")

from langchain import HuggingFaceHub
llm = HuggingFaceHub(repo_id = "google/flan-t5-xl")

# The LLM takes a prompt as an input and outputs a completion
prompt = "Alice has a parrot. What animal is Alice's pet?"
completion = llm(prompt)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Alternatively, open-source text embedding model hosted on Hugging Face
# pip install sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# The embeddings model takes a text as an input and outputs a list of floats
text = "Alice has a parrot. What animal is Alice's pet?"
text_embedding = embeddings.embed_query(text)

from langchain import PromptTemplate

template = "What is a good name for a company that makes {product}?"

prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)

prompt.format(product="colorful socks")