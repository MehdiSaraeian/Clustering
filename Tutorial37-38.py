import numpy as np
from sklearn import preprocessing
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''
Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival: Survival (0 = No; 1 = Yes)
name: Name
sex: Sex
age: Age
sibsp: Number of Siblings/Spouses Aboard
parch: Number of Parents/Children Aboard
ticket: Ticket Number
fare: Passenger Fare (British pound)
cabin: Cabin
embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat: Lifeboat
body: Body Identification Number
home.dest: Home/Destination
'''
'''
The K-Mean Algorithm:
    1. Choose value for K
    2. Randomly select K feature sets to start as your centroids
    3. Calculate distance of all other feature sets to centroids
    4. Classify other feature sets as same as closest centroid
    5. Take mean of each class (mean of all feature sets by class), making that mean the new centroid
    6. Repeat steps 3-5 until optimized (centroids no longer moving)
'''


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        for k in range(self.k):
            self.centroids[k] = data[k]
        for iteration in range(self.max_iter):
            for j in range(self.k):
                self.classifications[j] = []
            for FeatureSet in data:
                distances = [np.linalg.norm(FeatureSet - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(FeatureSet)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) > self.tol:
                    # print(abs(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)))
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        for featureSet in data:
            distances = [np.linalg.norm(featureSet - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification


def handle_non_numerical_data(data):
    # handling non-numerical data: must convert.
    columns = data.columns.values
    text_digit_val = {}

    def convert_to_int(val):
        return text_digit_val[val]
    for column in columns:
        # print(column, data[column].dtype)
        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            column_contents = data[column].values.tolist()
            # finding just the uniques
            unique_elements = set(column_contents)
            # great, found them.
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    # creating dict that contains new id per unique string
                    text_digit_val[unique] = x
                    x += 1
            # now we map the new "id" value to replace the string.
            data[column] = list(map(convert_to_int, data[column]))
    return data


# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'boat', 'home.dest', 'embarked', 'ticket'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
print(df.head(), '\n')
df.fillna(0, inplace=True)

df = handle_non_numerical_data(df)
# add/remove features just to see impact they have.
df.drop(['age'], 1, inplace=True)
print(df.head(), '\n')

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans()
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print('\n', correct/len(X))
