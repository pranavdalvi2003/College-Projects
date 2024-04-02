import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load your dataset
data = pd.read_csv('order_data.csv')

# Define your features (X) and the target variable (y)
X = data.drop('Rating', axis=1)
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=[str(i) for i in clf.classes_])
plt.tight_layout()

# Save the visualization to a bytes object
img_data = BytesIO()
plt.savefig(img_data, format='png')
img_data.seek(0)

# Define a route to display the decision tree visualization
@app.route('/decision_tree')
def decision_tree():
    img_base64 = base64.b64encode(img_data.read()).decode()
    return render_template('decision_tree.html', img_base64=img_base64)

# Define a route to make predictions and display the results
@app.route('/predict')
def predict():
    # Here, you can add code to make predictions on new data if needed.
    # For example, you can use clf.predict() on new data.

    # You can also analyze the results and make recommendations based on the demand for different products.
    # You can check which products have high ratings (indicating higher demand) and which ones have lower ratings.

    # For example, you can calculate the average rating for each product to identify which products are in higher demand.
    average_ratings = data.groupby('Order ID')['Rating'].mean()

    # Then, you can sort the products based on their average ratings to prioritize production.
    sorted_products = average_ratings.sort_values(ascending=False)

    # The products at the top of the list are in higher demand, so you may consider increasing their production.

    # You can also analyze other features in your dataset to make more informed production recommendations.

    products_in_demand = sorted_products.head().to_dict()

    return render_template('predictions.html', products_in_demand=products_in_demand)

if __name__ == '__main__':
    app.run(debug=True)
