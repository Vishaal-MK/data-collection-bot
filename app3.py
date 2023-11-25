from torch import cuda

import streamlit as st
from streamlit_chat import message

from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory


print(cuda.current_device())
st.set_page_config(page_title="Data Collection Chatbot")
st.header("Custom data collection chatbot")

@st.cache_resource()
def load_llm():

    llm = CTransformers(
        model='models\llama-2-7b-chat.Q2_K.gguf',
        model_type='llama',
        config={'temperature': 0.5, 'gpu_layers': 200, 'max_new_tokens': 256}
    )
    
    return llm

@st.cache_data()
def load_prompt_template():

    template = """[INST] <<SYS>>
        You're are a helpful Assistant, and you only respond to the "Assistant" tag. Remember, maintain a natural tone. Be precise, concise, and casual. Keep it short
        <</SYS>> 
        {chat_history}
      
        User:{user_response}
        
        Assistant:[/INST]
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['chat_history', 'user_response']
    )

    return prompt


def create_chain(memory):

    llm = load_llm()
    prompt = load_prompt_template()

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    return llm_chain

def generate_response(response, llm_chain):

    return llm_chain({'user_response': response})

def get_user_input():

    # get the user query
    input_text = st.text_input('Start talking!', "", key='input')
    return input_text


memory = ConversationBufferMemory(input_key='user_response', memory_key='chat_history', return_messages=True, k=6)
llm_chain = create_chain(memory=memory)


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# get the user query
user_input = get_user_input()


if user_input:

    # generate response to the user input
    response = generate_response(response=user_input, llm_chain=llm_chain)
    # memory.chat_memory.add_user_message(user_input)
    # memory.chat_memory.add_ai_message(response['text'])
    print('response:', response)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response['text'])


# display conversaion history (if there is one)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        