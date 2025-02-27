import streamlit as st
st.text_input("Your name", key="name")  # <-- Set the key of this text box to "name".

# You can access the value at any point with:
st.session_state.name