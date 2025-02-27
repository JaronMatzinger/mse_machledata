import streamlit as st

st.title("Parrot 🦜")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(
            message["content"]
        )  # st.markdown interprets and renders its input as markdown

# React to user input
if prompt := st.chat_input("Say something!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"{prompt}"

    # Display parrots response in chat message container
    with st.chat_message("🦜"):
        st.markdown(response)

    # Add parrot's response to chat history
    st.session_state.messages.append({"role": "parrot", "content": response})