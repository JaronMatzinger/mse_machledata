import streamlit as st

# Draw a title and some text to the app:
st.write('''
# This is the document title

This is some _markdown_.
This is some more **markdown**.
This is even more __markdown__.
''')

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3]})
st.write(df)  # 👈 Draw the dataframe

x = 10
st.write('x', x)  # 👈 Draw the string 'x' and then the value of x

# Also works with most supported chart types
import matplotlib.pyplot as plt
import numpy as np

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

st.write(fig)  # 👈 Draw a Matplotlib chart

from transformers import AutoModelForSequenceClassification
from torchinfo import summary

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
st.write(summary(model, input_size=(2, 512), dtypes=['torch.IntTensor']))