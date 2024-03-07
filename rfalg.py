import pandas as pd
import numpy as np
# from sklearn.preprocessing import Imputer from sklearn.ensemble import RandomForestClassifier,
# RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn import preprocessing

train_df = pd.read_csv(r".\trainDF.csv")

train_df['availability'] = preprocessing.LabelEncoder().fit_transform(train_df['availability'].values)

feature_cols = ['area_type', 'bath', 'balcony', 'Sqft_range_starts', 'Sqft_range_ends', 'Size', 'Flat_layout',
                'availability']
target = ['price']
X = train_df[feature_cols]  # Features
y = train_df[target]

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

print(train_features.shape)

print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

test_labels = np.array(test_labels['price'])
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
# Train the model on training data
rf.fit(train_features, train_labels.values.ravel())

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors


# print("Accuracy:",accuracy_score(test_labels, predictions))

errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("-------------Forward Fill---------------")
print("-------------RandomForest Regressor---------------")
print('Accuracy:', round(accuracy, 2), '%.')

df = pd.DataFrame(test_labels)
df.columns = ['test']
df2 = pd.DataFrame(predictions)
df2.columns = ['result']

df.insert(1, "Predict", df2, True)

df.plot(x='test', y='Predict', style='o')
plt.title('Regression')
plt.xlabel('Test')
plt.ylabel('Result')
plt.show()
