import pandas as pd
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split  
from sklearn import metrics  
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image

# Load dataset
pima = pd.read_csv("diabetes.csv")

# Print actual column names to check for discrepancies
print("Actual column names:", pima.columns)

# Use correct column names
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols]  # Features
y = pima['Outcome']  # Target variable

# Convert all features to numeric (if needed)
X = X.apply(pd.to_numeric, errors='coerce')

# Check for NaN values
if X.isnull().values.any():
    print("Warning: Some values are NaN after conversion!")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train the model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Print Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Visualize the decision tree
dot_data = StringIO()
export_graphviz(
    clf, out_file=dot_data,  
    filled=True, rounded=True,
    special_characters=True, feature_names=feature_cols, class_names=['0', '1']
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
