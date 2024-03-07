import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import tree, metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

train_df = pd.read_csv(r".\trainDF.csv")

# y.astype('category')
train_df['availability'] = preprocessing.LabelEncoder().fit_transform(train_df['availability'].values)
train_df['society'] = preprocessing.LabelEncoder().fit_transform(train_df['society'].values)
train_df['location'] = preprocessing.LabelEncoder().fit_transform(train_df['location'].values)

feature_cols = ['area_type', 'bath', 'balcony', 'Sqft_range_starts', 'Sqft_range_ends', 'Size', 'Flat_layout',
                'availability', 'society', 'location']
target = ['price']
X = train_df[feature_cols]  # Features
y = train_df[target]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 70% training and 30% test

from sklearn.feature_selection import RFE
# from sklearn.datasets import load_svmlight_file
# from array import array
from sklearn import linear_model

model = linear_model.LinearRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model)
rfe = rfe.fit(train_df[['area_type', 'Flat_layout', 'availability', 'location', 'society']], train_df['price'])
# score the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

model = linear_model.LinearRegression()
rfe = RFE(model)
rfe = rfe.fit(train_df[['bath', 'balcony', 'Sqft_range_starts', 'Sqft_range_ends', 'Size']], train_df['price'])
# score the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
print("----------------Forward Fill--------------------")
print("----------------Linear Regression--------------------")
model.fit(train_df[['area_type', 'bath', 'balcony', 'Sqft_range_starts', 'Sqft_range_ends', 'Size', 'Flat_layout',
                    'location', 'availability', 'society']], train_df['price'])
model.fit(X_train, y_train)
prediction = model.predict(X_test)
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, prediction)
print('mse', error)
z = (error / y_test.mean())
print("Accuracy :", 100 - 100 * z)
