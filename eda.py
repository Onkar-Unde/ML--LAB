# Name:Onkar Unde
# Roll:117
# Assingment No.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import date


warnings.filterwarnings('ignore')


script_dir = os.path.dirname(os.path.abspath(__file__))  
csv_path = os.path.join(script_dir, "used_cars_data.csv")  


if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    print("CSV file loaded successfully!\n")
else:
    print(f"Error: File not found at {csv_path}")
    exit()  


print("Top 5 Rows:\n", data.head())
print("\nLast 5 Rows:\n", data.tail())
print("\n Dataset Info:\n")
print(data.info())


print("\nUnique Values per Column:\n", data.nunique())


print("\nMissing Values Count:\n", data.isnull().sum())
print("\nMissing Values Percentage:\n", (data.isnull().sum() / len(data)) * 100)


if 'S.No.' in data.columns:
    data = data.drop(['S.No.'], axis=1)
    print("\nColumn 'S.No.' dropped.")

print("\nUpdated Dataset Info:\n", data.info())


if 'Year' in data.columns:
    data['Car_Age'] = date.today().year - data['Year']
    print("\nUpdated Data (with Car_Age column):\n", data.head())
else:
    print("Error: 'Year' column not found in dataset.")


    
if 'Car_Age' in data.columns and 'Price' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Car_Age'], y=data['Price'], alpha=0.6, color='b')

    plt.xlabel("Car Age (Years)")
    plt.ylabel("Price (in currency)")
    plt.title("Scatter Plot of Car Age vs Price")
    plt.grid(True)
    plt.show()
else:
    print("\nError: Columns 'Car_Age' or 'Price' not found in dataset.")


if 'Price' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Price'], color='skyblue')

    plt.xlabel("Price (in currency)")
    plt.title("Boxplot of Car Prices")
    plt.grid(True)
    plt.show()
else:
    print("\n Error: 'Price' column not found in dataset.")
