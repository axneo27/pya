import numpy as np
import pandas as pd

# Python list
py_list = [1, 2, 3]
print("Python list * 2:", py_list * 2)  # [1, 2, 3, 1, 2, 3]

# NumPy array
np_array = np.array([1, 2, 3])
print("NumPy array * 2:", np_array * 2)  # [2 4 6]

# Series
data = [10, 20, 30]
s = pd.Series(data, index=['a', 'b', 'c'])
print(s)

# DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
}
df = pd.DataFrame(data)
print(df)

# csv read
df = pd.read_csv('./assets/apartments.csv')

df['age'] = 2026 - df['year_built'] # computed column

df = df.sort_values(by='age', ascending=False) # sort by age

filtered_df = df[df['age'] > 10]
print("Filtered: ", filtered_df)

# csv write
df.to_csv('./assets/apartaments_output.csv', index=False)

# methods
print("Head:")
print(df.head())

print("\nTail:")
print(df.tail())

print("\nSample:")
print(df.sample(1))

print("\nInfo:")
df.info()

print("\nDescribe:")
print(df.describe())

# desribe: 20x5 22.5 25 27.5 30x1
