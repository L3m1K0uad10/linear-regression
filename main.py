import numpy as np
import pandas as pd



# Real estate dataset
df = pd.read_csv("dataset/Real_estate.csv")
data = df.to_numpy()

rows, cols = data.shape

## preprocessing

# remove the column "No" which is at 0
data = data[0:rows + 1, 1:cols + 1]
print(data[0:2])