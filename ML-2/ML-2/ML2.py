import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df_sal = pd.read_csv(r"C:\Users\Icon\Downloads\ML-2\ML-2\Salary_Data.csv")  # Use full path if needed

# Display first rows and summary
print(df_sal.head())  
print(df_sal.describe())  

# Salary Distribution Plot
plt.figure(figsize=(7, 5))
plt.title('Salary Distribution Plot')
sns.histplot(df_sal['Salary'], kde=True, color='lightcoral')  # Fixed deprecated function
plt.show()

# Scatter plot (Salary vs Experience)
plt.figure(figsize=(7, 5))
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color='lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid(True)  # Add grid for better visualization
plt.show()

# Splitting variables
X = df_sal[['YearsExperience']]  # Independent variable
y = df_sal[['Salary']]  # Dependent variable

# Splitting dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predictions
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

# Plot Regression Line on Training Set
plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, color='lightcoral', label="Training Data")
plt.plot(X_train, y_pred_train, color='firebrick', label="Regression Line")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Plot Regression Line on Test Set
plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, color='lightblue', label="Test Data")
plt.plot(X_train, y_pred_train, color='firebrick', label="Regression Line")  # Keep the line same as training
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Print Model Coefficients
print(f'Coefficient: {regressor.coef_[0][0]}')
print(f'Intercept: {regressor.intercept_[0]}')
