import pandas as pd

from sklearn import preprocessing

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor,
# GradientBoostingRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(r".\trainDF.csv")

# bins=[0,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120,135,150,200,300,4000]
# group_names=['41','42','43','44','45','46','47','48','49','50','52','53','54','56','60','72','85','98','110','124','148','160']

# train_df['price_range']=pd.cut(train_df['price'],bins,labels=group_names)

# train_df['price_range']= train_df['price_range'].apply(pd.to_numeric, errors='coerce')
train_df['availability'] = preprocessing.LabelEncoder().fit_transform(train_df['availability'].values)
# train_df['society'] = preprocessing.LabelEncoder().fit_transform(train_df['society'].values)
# train_df['location'] = preprocessing.LabelEncoder().fit_transform(train_df['location'].values)

feature_cols = ['area_type', 'bath', 'balcony', 'Sqft_range_starts', 'Sqft_range_ends', 'Size', 'Flat_layout',
                'availability']
target = ['price']
X = train_df[feature_cols]  # Features
y = train_df[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # 70% training and 30% test

from sklearn import svm

clf = svm.SVR(kernel='linear')  # Linear Kernel

clf.fit(x_train, y_train.values.ravel())

y_pred = clf.predict(x_test)

# from sklearn import metrics
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, y_pred)
z = (error / y_test.mean())
print("------------------Forward Fill-----------------")

print("------------------SVM Regressor-----------------")
print("Accuracy :", 100 - 100 * z)
