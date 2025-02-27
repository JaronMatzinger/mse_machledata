# Building a chatbot

Streamlit has dedicate UI elements for chatbots:

- `st.chat_message`, which represents a chat message.
- `st.chat_input`, i.e. a text box to type messages into.

You can find the docs for the chat elements in the [API reference](https://docs.streamlit.io/library/api-reference/chat).

Crucially, you can combine these elements with the session state track the chat history. If you haven't read it yet, now is the time to learn more about [session state](https://docs.streamlit.io/library/advanced-features/session-state).

Here's a little demo of the chat UI elements. This "chatbot" simply repeats the message you sent.

```python
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
```

You can find this snippet in `snippets/parrot_bot.py`

## Streaming responses

To get the streaming effect from ChatGPT we all know and love, you can use `st.write_stream` together with a [python generator](https://realpython.com/introduction-to-python-generators/):

```python
import streamlit as st

def message_generator(message: str):
    for char in message:
        yield char

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
        st.write_stream(response)

    # Add parrot's response to chat history
    st.session_state.messages.append({"role": "parrot", "content": response})
```

Why is this useful? Text generation takes time. Because humans are impatient, it is almost always a good idea to keep them busy! In the Hugging Face `transformers` library, you can use the [the `TextStreamer` class](https://huggingface.co/docs/transformers/v4.37.2/en/internal/generation_utils#transformers.TextStreamer) to stream the output of the model!

## Your turn: Build a _real_ chatbot

Use the knowledge you've gained today to build a chatbot powered by an LLM from the Hugging Face Hub. Use the bot you created in [`notebooks/transformers.ipynb`](./notebooks/transformers.ipynb) as a starting point!

Don't forget that the `transformers` library comes with [utilities for chat models](https://huggingface.co/docs/transformers/main/chat_templating#advanced-template-writing-tips)!

### Going further

Here are some ideas to enhance your app:

- Add a `st.selectbox` to change the model.
- Learn about text generation strategies [here](https://huggingface.co/docs/transformers/main/generation_strategies#text-generation-strategies) and add options to the UI to change the generation settings.
- Depending on the hardware available to you use enhanced [CPU](https://huggingface.co/docs/transformers/main/en/perf_infer_cpu) or [GPU](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one) inference techniques so run bigger models!
