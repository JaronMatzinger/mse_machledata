import transformers
import streamlit as st

@st.cache_resource
def load_model():
    model = transformers.pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B",
        max_new_tokens=100
    )
    instruction = ("You are a helpful chatbot. Answer concisely and accurately.")
    return model, instruction

model, instruction = load_model()

def generate_response(question):
    prompt = f'{instruction}\nUser: {question}\nBot:'
    response = model(prompt)[0]['generated_text']
    return response.split('Bot:')[-1].strip()

st.title('HF Chatbot 🤖')

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if user_input := st.chat_input('Ask me anything!'):
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(user_input)

    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Process the user input
    response = generate_response(user_input)

    # Show model response
    with st.chat_message('bot'):
        st.markdown(response)

    # Save chat history
    st.session_state.messages.append({'role': 'bot', 'content': response})
    