import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'boat', 'home.dest', 'embarked', 'ticket'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head(), '\n')


def handle_non_numerical_data(data):
    columns = data.columns.values

    for column in columns:
        text_digit_val = {}

        def convert_to_int(val):
            return text_digit_val[val]

        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            column_contents = data[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1
            data[column] = list(map(convert_to_int, data[column]))
    return data


df = handle_non_numerical_data(df)
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    print(predict_me)
    predict_me = predict_me.reshape(-1, len(predict_me))
    print(predict_me)
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
