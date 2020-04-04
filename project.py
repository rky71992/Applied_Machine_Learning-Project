#
#importing package dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



#Preparing Data
#from dataset_file, extract #1Reg No, Course, Grades and Branch = Food Science and Technology
initial_dataset = pd.read_csv("dataset_file.csv",usecols=[0,1,2,3,4,5,6,8,9])
#initial_dataset contains all rows, extracting rows with 'MHRDName' == 'Master of Science (Food Science and Technology)'
print("Total number of Rows loaded:",initial_dataset.shape)
corrected_dataset = initial_dataset[initial_dataset['MHRDName'] == 'Master of Science (Food Science and Technology)']
print("Rows of MHRDName =Master of Science Food Science and Technology:",corrected_dataset.shape)
corrected_dataset = corrected_dataset.drop('MHRDName',axis=1)
corrected_dataset = corrected_dataset.dropna(axis=0)
print("Rows after droping NaNa rows:",corrected_dataset.shape)
print("Colums headings:\n",corrected_dataset.columns.values)

labelEncoder = LabelEncoder()
labelEncoder.fit(corrected_dataset['Grade'])
corrected_dataset['Grade'] = labelEncoder.transform(corrected_dataset['Grade'])

#dividing into test train
train, test = train_test_split(corrected_dataset, test_size=0.2)
#print("Test shape:",test.shape,"Training Shape:",train.shape)
#x = df.iloc[:, [0,1,2,3]].values
#SInce its a clustring algorithm we are droping grade column from training dataset
train = train.drop(['Grade','Termid','Regd No','Course'],axis=1)
#kmeans = KMeans(n_clusters=4)
#kmeans.fit(corrected_dataset)
print(corrected_dataset.head())
x = corrected_dataset.iloc[:,[4,5,6,7]].values
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(x)
y_kmeans5 = kmeans.fit_predict(x)
print(y_kmeans5)
print(kmeans.cluster_centers_)
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
#import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
#plt.scatter(train[:,0],train[:,1])

'''correct = 0
for i in range(len(test)):
    predict_me = np.array(test[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == test[i]:
        correct += 1

print(correct/len(test))'''