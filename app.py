import re
import os
import spacy
import GPUtil
import pandas as pd
from transformers import pipeline

import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_experimental.prompt_injection_identifier import  HuggingFaceInjectionIdentifier


# App title
st.set_page_config(page_title="💬 LLama2 Langchain ChatBot")

# Setup NER
nlp = spacy.load("en_core_web_lg")
ner_categories = ['PERSON', 'DATE']
email_extraction_pattern = r"\S+@\S+\.\S+"
phone_extraction_pattern = r"^[0-9]{10}$"

# Add sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", device=-1, model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Load injection identifier model
injection_identifier = HuggingFaceInjectionIdentifier()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! May I know your name?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


# Make sure the model path is correct for your system!
if GPUtil.getGPUs():
    llm = CTransformers(
            model='models\llama-2-7b-chat.Q2_K.gguf',
            model_type='llama',
            config={'temperature': 0.5, 'gpu_layers': 200, 'max_new_tokens': 70, 'context_length': 1024}
        )
else:
    llm = CTransformers(
            model='models\llama-2-7b-chat.Q2_K.gguf',
            model_type='llama',
            config={'temperature': 0.5, 'gpu_layers': 0, 'max_new_tokens': 70, 'context_length': 1024}
        )


# Function for generating LLM response based on sentiment
def generate_response(prompt_input, chat_history, sentiment):

    if sentiment == "NEGATIVE":
        template = """[INST] <<SYS>>
            You are chatbot whose goal is to assure users that their data is safe and secure. Engage in small talk until the user is willing to share thier information. Don't keep asking the same questions. Keep your replies and questions to a maximum 15 words.
            <</SYS>>

            Hi there! [/INST] 
            Hello! May I know your name?
            {history}
            [INST] {human_input}
        """
    else:
        template = """[INST] <<SYS>>
            You are chatbot whose goal is to collect user's name, email, date-of-birth and phone-number. You should operate as a persuasive conversationalist, encouraging users to share their information willingly. Don't keep asking the same questions. Keep your replies and questions to a maximum 15 words.
            <</SYS>>

            Hi there! [/INST] 
            Hello! May I know your name?
            {history}
            [INST] {human_input}
        """

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    chatbot = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        # memory=ConversationSummaryMemory.from_messages(llm=llm, memory_key="history", chat_memory=chat_history),
        memory=ConversationBufferMemory(memory_key="history", chat_memory=chat_history, ai_prefix='', human_prefix='[INST]'),
    )
    return chatbot.predict(human_input=f"{prompt_input} [\INST]")


# Extract user information if given
def extract_entities(prompt, doc):
    default_data = {
        'Name': None,
        'Date of Birth': None,
        'Email': None,
        'Phone Number': None
    }

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            default_data['Name'] = ent.text
        elif ent.label_ == "DATE":
            default_data['Date of Birth'] = ent.text
    
    if re.findall(email_extraction_pattern, prompt):
        default_data["Email"] = re.findall(email_extraction_pattern, prompt)[0]
    elif re.match(phone_extraction_pattern, prompt):
        default_data["Phone Number"] = re.findall(phone_extraction_pattern, prompt)[0]

    update_database(default_data)


# Function to update dummy database
def update_database(data):
    if os.path.isfile('database/info.csv'):
        df = pd.read_csv('database/info.csv')
        df.update([data])
        df.to_csv('database/info.csv', index=False)
    else:
        os.mkdir('database')
        df = pd.DataFrame.from_dict([data])
        df.to_csv('database/info.csv', index=False)


# Generate a new response if last message is not from assistant
chat_history = StreamlitChatMessageHistory()


if st.session_state.messages[-1]["role"] != "assistant":
    doc = nlp(prompt)
    sentiment = sentiment_pipeline(prompt)[0]
    try:
        injection_identifier.run(prompt)
        extract_entities(prompt, doc)
        with st.chat_message("assistant"):
            with st.spinner():
                response = generate_response(prompt, chat_history, sentiment) 
                st.write(response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
    
    except:
        with st.chat_message("assistant"):
            with st.spinner():
                response = "Prompt Injection Detected"
                st.write(response)