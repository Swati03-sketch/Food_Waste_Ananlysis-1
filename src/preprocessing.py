import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/Sony/Documents/global_food_wastage_dataset.csv")
df.head()
df.isnull().sum().sort_values()
df.shape
df.describe()
df.dtypes
