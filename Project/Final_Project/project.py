# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import csv

# %%
"""
### Pre-Processing for Train Set
"""

# %%
from sklearn import preprocessing
# Preprocessing for Train set
data_train = pd.read_csv("train.csv")
column_means = data_train.mean()
data_train = data_train.fillna(column_means)

# Converting Columns of string value into integers
data_train['Gender'] = data_train['Gender'].replace(to_replace = ['Male', 'Female'], value = ['0', '1'])
data_train['Customer Type'] = data_train['Customer Type'].replace(to_replace = ['Loyal Customer', 'disloyal Customer'], value = ['0', '1'])
data_train['Type of Travel'] = data_train['Type of Travel'].replace(to_replace = ['Personal Travel', 'Business travel'], value = ['0', '1'])
data_train['Class'] = data_train['Class'].replace(to_replace = ['Eco Plus', 'Business', 'Eco'], value = ['0', '1', '2'])
data_train['satisfaction'] = data_train['satisfaction'].replace(to_replace = ['neutral or dissatisfied', 'satisfied'], value = ['0', '1'])

# Normalize dataset
scaler = preprocessing.MinMaxScaler()
names = data_train.columns
d = scaler.fit_transform(data_train)
data_train = pd.DataFrame(d, columns=names)

x_train = data_train.iloc[0:,2:24]  #independent columns
y_train = data_train.iloc[0:,24]    #target column i.e price range

# %%
"""
### Pre-Processing for Test Set
"""

# %%
# Preprocessing for Test set
data_test = pd.read_csv("test.csv")
column_means = data_test.mean()
data_test = data_test.fillna(column_means)

# Converting Columns of string value into integers
data_test['Gender'] = data_test['Gender'].replace(to_replace = ['Male', 'Female'], value = ['0', '1'])
data_test['Customer Type'] = data_test['Customer Type'].replace(to_replace = ['Loyal Customer', 'disloyal Customer'], value = ['0', '1'])
data_test['Type of Travel'] = data_test['Type of Travel'].replace(to_replace = ['Personal Travel', 'Business travel'], value = ['0', '1'])
data_test['Class'] = data_test['Class'].replace(to_replace = ['Eco Plus', 'Business', 'Eco'], value = ['0', '1', '2'])
data_test['satisfaction'] = data_test['satisfaction'].replace(to_replace = ['neutral or dissatisfied', 'satisfied'], value = ['0', '1'])

# Normalize dataset
scaler = preprocessing.MinMaxScaler()
names = data_test.columns
d = scaler.fit_transform(data_test)
data_test = pd.DataFrame(d, columns=names)

x_test = data_test.iloc[0:,2:24]  #independent columns
y_test = data_test.iloc[0:,24]    #target column i.e price range

# %%
"""
## 1. Univariate Selection
"""

# %%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=22)
fit = bestfeatures.fit(x_train,y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x_train.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(22,'Score'))  #print 22 features

# %%
"""
## 2. Feature Importance
"""

# %%
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(x_train,y_train)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x_train.columns)
feat_importances.nlargest(22).plot(kind='barh')
plt.show()

# %%
"""
## 3.Correlation Matrix with Heatmap
"""

# %%
import seaborn as sns

#get correlations of each features in dataset
corrmat = data_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# %%
"""
### Decision Tree
"""

# %%
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

begin_time = time.time()
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
end_time = time.time()
time_taken = end_time - begin_time
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("time taken for the Decision Tree classifier:",time_taken)

# %%
"""
### KNN
"""

# %%
# KNN
from sklearn.neighbors import KNeighborsClassifier

begin_time = time.time()

neigh = KNeighborsClassifier(n_neighbors=3)
neigh = neigh.fit(x_train, y_train)

y_pred_knn = neigh.predict(x_test)
end_time = time.time()
time_taken = end_time - begin_time

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
print("time taken for the KNN classifier:",time_taken)

# %%
"""
### Support Vector Machine
"""

# %%
# Support Vector Machine
from sklearn.svm import SVC
begin_time = time.time()
svclassifier = SVC(kernel='rbf')
svc = svclassifier.fit(x_train, y_train)
y_prediction = svc.predict(x_test)
end_time = time.time()
time_taken = end_time - begin_time

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_prediction))
print("time taken for the SVM classifier:",time_taken)

# %%
"""
### Random Forest
"""

# %%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
begin_time = time.time()
rclf=RandomForestClassifier(n_estimators=100)
rclf = rclf.fit(x_train,y_train)
y_pred_rf = rclf.predict(x_test)
end_time = time.time()
time_taken = end_time - begin_time
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))
print("time taken for the Random Forest classifier:",time_taken)

# %%
"""
### Creating new Training & Testing dataset based on the Best Features
"""

# %%
# Create new training & testing dataset based on best features
remove_features = {'Gate location', 'Gender', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Departure/Arrival time convenient', 'Age', 'Inflight service', 'Food and drink', 'Checkin service', 'Flight Distance'}
x_train_feature = x_train.drop(remove_features,axis=1)
x_test_feature = x_test.drop(remove_features,axis=1)
print(x_train_feature)


# %%
"""
### Decision Tree for Best Features
"""

# %%
# Decision Tree for best features
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
begin_time = time.time()
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train_feature,y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test_feature)

end_time = time.time()
time_taken = end_time - begin_time

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("time taken for the Decision Tree classifier:",time_taken)

# %%
"""
### KNN for the Best Features
"""

# %%
# KNN for best features
from sklearn.neighbors import KNeighborsClassifier
begin_time = time.time()
neigh = KNeighborsClassifier(n_neighbors=3)
neigh = neigh.fit(x_train_feature, y_train)

y_pred_knn = neigh.predict(x_test_feature)
end_time = time.time()
time_taken = end_time - begin_time
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
print("time taken for the KNN classifier:",time_taken)

# %%
"""
### Support Vector MAchine for the Best Features
"""

# %%
# Support Vector Machine for best features
from sklearn.svm import SVC
begin_time = time.time()
svclassifier = SVC(kernel='rbf')
svclassifier = svclassifier.fit(x_train_feature, y_train)
y_pred_svm = svclassifier.predict(x_test_feature)
end_time = time.time()
time_taken = end_time - begin_time
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
print("time taken for the SVM classifier:",time_taken)

# %%
"""
### Random Forest for the Best Features
"""

# %%
# Random Forest for best features
from sklearn.ensemble import RandomForestClassifier
begin_time = time.time()
rclf=RandomForestClassifier(n_estimators=100)
rclf = rclf.fit(x_train_feature,y_train)
y_pred_rf = rclf.predict(x_test_feature)
end_time = time.time()
time_taken = end_time - begin_time
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))
print("time taken for the Random Forest classifier:", time_taken)
# %%
