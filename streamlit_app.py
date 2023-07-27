import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

st.title('🎈 Text summarization app')

import sys
st.write(sys.executable)

text = st.text_area('Enter some text to summarize')

def generate_response(text):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = TokenTextSplitter(chunk_size= 1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in chunks]
    chain = load_summarize_chain(llm=llm,chain_type="map_reduce")
    return chain.run(docs)

with st.form('my_form'):
    openai_api_key = st.text_input('Enter your OpenAI API key:', type='password')
    submitted = st.form_submit_button('Submit')

    if submitted & (not openai_api_key.startswith('sk-')):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted & openai_api_key.startswith('sk-'):
        st.info(generate_response(text))
