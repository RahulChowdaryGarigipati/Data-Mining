import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier


############################## Data Visualization & Analysis #############################
data=pd.read_csv('train.csv')
column_means = data.mean()
data = data.fillna(column_means)
sns.countplot(x='satisfaction',data=data)
plt.title('Number of passengers that are satified or neutral & dissatisfied')
plt.show()
sns.countplot(x='Class',data=data)
plt.title('Number of passengers in each Class')
plt.show()
loyal = data[data['Customer Type'] == 'Loyal Customer']
disloyal = data[data['Customer Type'] == 'disloyal Customer']
sns.countplot(x='satisfaction', data=loyal)
plt.title('Number of passengers among loyal customers that are satified or neutral & dissatisfied')
plt.show()
sns.countplot(x='satisfaction', data=disloyal)
plt.title('Number of passengers among disloyal customers that are satified or neutral & dissatisfied')
plt.show()
businessclass = data[data['Class'] == 'Business']
ecoplus = data[data['Class'] == 'Eco Plus']
economy = data[data['Class'] == 'Eco']
# print(len(businessclass.index))
# print(len(ecoplus.index))
# print(len(economy.index))
# print(type(businessclass.mean()))
bcreviews = businessclass.iloc[:,8:22]
ar=bcreviews.mean()
# print(ar.mean())
epreviews = ecoplus.iloc[:,8:22]
ar1=epreviews.mean()
# print(ar1.mean())
ereviews = economy.iloc[:,8:22]
ar2 = ereviews.mean()
# print(ar2.mean())
fclass= ['Business','Eco Plus', 'Eco']
rating = [ar.mean(),ar1.mean(),ar2.mean()]
fdistance = data['Flight Distance']
plt.plot(fclass, rating)
plt.title('Comparison of class-vise overall rating')
plt.xlabel('Class')
plt.ylabel('Overall Rating')
plt.show()
# plt.plot(fclass,fdistance)
# plt.title('Distance and class')
# plt.xlabel('Class')
# plt.ylabel('Flight distance')
# plt.show()
btravel = data[data['Type of Travel'] == 'Business travel']
ptravel = data[data['Type of Travel'] == 'Personal Travel']
npbb = (btravel['Class']=='Business').sum()
npepb = (btravel['Class']=='Eco Plus').sum()
npeb = (btravel['Class']=='Eco').sum()
c = ['Business','Eco Plus', 'Eco']
npbpc = [npbb,npepb,npeb]
plt.plot(c, npbpc)
plt.title('Number of people in each class while Business Travel')
plt.xlabel('Class')
plt.ylabel('Number of people')
plt.show()
npbp = (ptravel['Class']=='Business').sum()
npepp = (ptravel['Class']=='Eco Plus').sum()
npep = (ptravel['Class']=='Eco').sum()
npppc = [npbp,npepp,npep]
plt.plot(c, npppc)
plt.title('Number of people in each class while Personal Travel')
plt.xlabel('Class')
plt.ylabel('Number of people')
plt.show()

############################ Classification Algorithms#################################

x = data.iloc[:, 7:24].values
y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# instantiate the model (using the default parameters)
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

# fit the model with data
pipe_knn.fit(x_train, y_train)

# apply scaling on testing data, without leaking training data.
pipe_knn.score(x_test, y_test)

y_pred_knn = pipe_knn.predict(x_test)

print("Accuracy KNN:", accuracy_score(y_test, y_pred_knn))
print("Precision KNN:", precision_score(y_test, y_pred_knn, average='weighted'))
print("Recall KNN:", recall_score(y_test, y_pred_knn, average='weighted'))

pipe_lg = make_pipeline(StandardScaler(), LogisticRegression())

# fit the model with data
pipe_lg.fit(x_train, y_train)

# apply scaling on testing data, without leaking training data.
pipe_lg.score(x_test, y_test)

y_pred_lg = pipe_lg.predict(x_test)
print("Accuracy Logistic Regression:", accuracy_score(y_test, y_pred_lg))
print("Precision Logistic Regression:", precision_score(y_test, y_pred_lg, average='weighted'))
print("Recall Logistic Regression:", recall_score(y_test, y_pred_lg, average='weighted'))
