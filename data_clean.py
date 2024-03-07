import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv(".\Predicting-House-Prices-In-Bengaluru-Train-Data.csv")

# print(df_train.head())
# print(df_train.describe())

print(df_train.isnull().sum())
# print(df_train['bath'].mean())
# print(len(df_train['availability'].value_counts()))
print(df_train['society'].mode())
# df_train['location'].isnull().dropna()
# df_train['society'] =df_train['society'].fillna("unknown")

df_train1 = df_train.fillna(method="ffill")
df_train2 = df_train.fillna(method="bfill")
df_train3 = df_train.fillna(df_train.mode().iloc[0])

# print(df_train.isnull().sum())

# print(df_train.tail())

# print(df_train[df_train["bath"].isnull()])


df_train1.to_csv(".\TrainFFill.csv")
df_train2.to_csv(".\TrainBFill.csv")
df_train3.to_csv(".\TrainMFill.csv")

# print(df_train['size'].value_counts())

sq = df_train['location'].value_counts()

# print(len(sq))

categorical_list1 = []
numerical_list1 = []
for i in df_train1.columns.tolist():
    if df_train1[i].dtype == 'object':
        categorical_list1.append(i)
    else:
        numerical_list1.append(i)

print('Number of categorical features:', str(len(categorical_list1)))
print('Number of numerical features:', str(len(numerical_list1)))
# print(numerical_list1)


# print(df_train['society'].isnull())

# print(df_train['society'].mode())


num = LabelEncoder()

df_train1['area_type'] = num.fit_transform(df_train1['area_type'].astype('str'))

# print(df_train1[df_train1['total_sqft']=="1550 - 1590"])

new = df_train1['total_sqft'].str.split("-", n=1, expand=True)
df_train1['Sqft_range_starts'] = new[0]
df_train1['Sqft_range_ends'] = new[1]

# print(new)
# print(df_train1['area_type'].value_counts())


m = df_train1[df_train1['Sqft_range_starts'].str.contains("Sq. Meter")]
n = m['Sqft_range_starts'].str.replace('Sq. Meter', '')

y = df_train1[df_train1['Sqft_range_starts'].str.contains("Sq. Yards")]
z = y['Sqft_range_starts'].str.replace('Sq. Yards', '')

a = df_train1[df_train1['Sqft_range_starts'].str.contains("Acres")]
b = a['Sqft_range_starts'].str.replace('Acres', '')

c = df_train1[df_train1['Sqft_range_starts'].str.contains("Cents")]
d = c['Sqft_range_starts'].str.replace('Cents', '')

e = df_train1[df_train1['Sqft_range_starts'].str.contains("Grounds")]
f = e['Sqft_range_starts'].str.replace('Grounds', '')

p = df_train1[df_train1['Sqft_range_starts'].str.contains("Guntha")]
q = p['Sqft_range_starts'].str.replace('Guntha', '')

r = df_train1[df_train1['Sqft_range_starts'].str.contains("Perch")]
s = r['Sqft_range_starts'].str.replace('Perch', '')

obj_df = df_train1.select_dtypes(include=['object']).copy()
# print(obj_df.head())
# print(pd.get_dummies(df_train1['area_type'].head(5)))


newsize = df_train1['size'].str.split(" ", n=1, expand=True)
df_train1['Size'] = newsize[0]
df_train1['Flat_layout'] = newsize[1]
# print(newsize)


# df_train1['Flat_layout']=pd.get_dummies(df_train1['Flat_layout'])
# print(bh)

df_train1['Flat_layout'] = num.fit_transform(df_train1['Flat_layout'].astype('str'))

# pd.to_numeric(df_train1['Sqft_range_starts'], errors='coerce')

# df_train1['Sqft_range_starts'].convert_objects(convert_numeric=True).dtypes
# df_train1['Sqft_range_starts']=df_train1['Sqft_range_starts'].apply(pd.to_numeric, errors='coerce').dtypes


# df_train1['Size'].convert_objects(convert_numeric=True).dtypes

# df_train1['Size']=df_train1['Size'].apply(pd.to_numeric, errors='coerce').dtypes

# df_train1['Size'].astype(str).astype(int)
df_train1['Size'] = df_train1['Size'].apply(pd.to_numeric, errors='coerce')
df_train1['Sqft_range_ends'] = df_train1['Sqft_range_ends'].apply(pd.to_numeric, errors='coerce')
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].apply(pd.to_numeric, errors='coerce')

'''
categorical_list = []
numerical_list = []
for i in df_train1.columns.tolist():
    if df_train1[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
        
print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))
print(numerical_list)
print(df_train1['Sqft_range_starts'].isnull().sum())

n['Sqft_range_starts'].str.contains("True")
foo = n.loc[n=="True"]
sns.countplot(n,label="count")
plt.show()
'''

sns.countplot(df_train1['Size'], label="count")
plt.show()
# print(df_train1['Flat_layout']=="2".sum())
result = pd.concat([df_train1, m], axis=1, join='inner')

n = n.apply(pd.to_numeric, errors='coerce')
SqMeter = n.apply(lambda x: x * 10.764)

z = z.apply(pd.to_numeric, errors='coerce')
SqYards = z.apply(lambda x: x * 9)

b = b.apply(pd.to_numeric, errors='coerce')
Acres = b.apply(lambda x: x * 43560)

d = d.apply(pd.to_numeric, errors='coerce')
Cents = d.apply(lambda x: x * 435.6)

f = f.apply(pd.to_numeric, errors='coerce')
Grounds = f.apply(lambda x: x * 2400)

q = q.apply(pd.to_numeric, errors='coerce')
Guntha = q.apply(lambda x: x * 1089)

s = s.apply(pd.to_numeric, errors='coerce')
Perch = s.apply(lambda x: x * 272.25)

# print(df_train['Sqft_range_starts'].isnull().sum())


pd.to_numeric(df_train1['Sqft_range_starts'], errors='coerce')
pd.to_numeric(df_train1['Sqft_range_starts'], errors='coerce').isnull()
df = df_train1[pd.to_numeric(df_train1['Sqft_range_starts'], errors='coerce').isnull()]

'''
df1_acres=pd.DataFrame({'Index':Acres.index, 'Sqft_range_starts':Acres.values})
df1_acres.drop(['Index'],axis=1)
df1_cents=pd.DataFrame({'Index':Cents.index, 'Sqft_range_starts':Cents.values})
df1_grounds=pd.DataFrame({'Index':Grounds.index, 'Sqft_range_starts':Grounds.values})
df1_guntha=pd.DataFrame({'Index':Guntha.index, 'Sqft_range_starts':Guntha.values})
df1_perch=pd.DataFrame({'Index':Perch.index, 'Sqft_range_starts':Perch.values})
df1_sqm=pd.DataFrame({'Index':SqMeter.index, 'Sqft_range_starts':SqMeter.values})
df1_sqy=pd.DataFrame({'Index':SqYards.index, 'Sqft_range_starts':SqYards.values})

'''

# Acres.to_frame(name=Acres)
# Acre=pd.concat([df_train1,Acres], axis=1, join='inner')
# Acres = pd.merge(df_train1, Acres, right_index=True, left_index=True,how ='left')
# d_acres = pd.DataFrame([Acres])

df1_acres = pd.DataFrame(Acres)
df1_cents = pd.DataFrame(Cents)
df1_grounds = pd.DataFrame(Grounds)
df1_guntha = pd.DataFrame(Guntha)
df1_perch = pd.DataFrame(Perch)
df1_sqm = pd.DataFrame(SqMeter)
df1_sqy = pd.DataFrame(SqYards)

df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_acres['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_cents['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_grounds['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_guntha['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_perch['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_sqm['Sqft_range_starts'])
df_train1['Sqft_range_starts'] = df_train1['Sqft_range_starts'].fillna(df1_sqy['Sqft_range_starts'])

df_train1['Sqft_range_ends'] = df_train1['Sqft_range_ends'].fillna(df_train1['Sqft_range_starts'])

train_df = df_train1.drop(['size', 'total_sqft'], axis=1)

train_df.to_csv(".\TrainDF.csv")

print(train_df.isnull().sum())

categorical_list = []
numerical_list = []
for i in train_df.columns.tolist():
    if train_df[i].dtype == 'object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)

print('Number of categorical features:', str(len(categorical_list)))
print('Number of numerical features:', str(len(numerical_list)))

#  print(df_train1[df_train1['total_sqft']=="1550 - 1590"])
