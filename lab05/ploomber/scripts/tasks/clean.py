import pandas as pd

upstream = ['raw']
product = None

df = pd.read_csv(upstream['raw']['data'])

# Data cleaning code goes here

df.to_csv(product['data'], index=False)