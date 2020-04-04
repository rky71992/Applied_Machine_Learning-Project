#importing packages dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler

#Preparing Data
#Only columns are loaded which are required for process
df = pd.read_csv("dataset_file.csv",usecols=[0,1,2,3,4,5,6,8,9])
print("Shape of data loaded:",df.shape)
print("Datahead:\n",df.head())

#Only rows are selected where MHRDName == Master of Science (Food Science and Technology)
df = df[df['MHRDName'] == 'Master of Science (Food Science and Technology)']
print("Rows of MHRDName = Master of Science (Food Science and Technology):",df.shape[0])
df = df.drop('MHRDName',axis=1)

#Checking for missing values rows, and droping missing value rows in this case
print("Data with NaNa value:\n",df.isnull().sum())
df = df.dropna(axis=0)
print("Data after removing NaNa value:\n",df.isnull().sum())
print("Rows after droping NaNa rows:",df.shape[0])

#Adding Labelencoders, generating uid for identification, droping processed columns and adding unique student_id
df['student_id'] = df["Termid"].astype(str) + "" + df["Regd No"].astype(str) + "" + df["Course"].astype(str)
#Trimimg unwanted columns
df = df.drop(['Termid','Regd No','Course'], axis=1)

#adding label encoder to grades for processing
unique_grades = len(df['Grade'].unique())
le = LabelEncoder()
df['Grade'] = le.fit_transform(df['Grade'].values)
print('After pre-processing Step 4\n',df.head())

#Clustring Techniques
#Kmeans: Unsupervised learning

kmeans_df = df.copy()
kmeans_df = df.drop(['student_id','Grade'],axis=1)
print("Columns in KMeans data input\n",kmeans_df.columns.values)
c = ['red','blue','green','yellow','orange','black','indigo']
distortion=[]

for i in range(2,unique_grades):
    kmeans = KMeans(n_clusters=i)
    y_km = kmeans.fit_predict(kmeans_df)
    print("Centers",kmeans.cluster_centers_)
    distortion.append(kmeans.inertia_)
    print('Silhouette score for {} cluster ='.format(i),silhouette_score(kmeans_df,y_km))
    kmeans_df_array = np.array(kmeans_df)
    for j in range(len(kmeans.cluster_centers_)):
        plt.scatter(kmeans_df_array[y_km==j,0],kmeans_df_array[y_km==j,1],marker='o',c=c[j])
    plt.title('KMeans Clustering with {} clusters'.format(i))
    plt.show()
print('Distortion-->',distortion)

a=np.arange(2,unique_grades)
plt.plot(a,distortion)
plt.grid()
plt.show()


#Agglomerative Clustering technique
ac_df = df.copy()
ac_df = df.drop(['student_id','Grade'],axis=1)
for i in range(2,4):
    ac = AgglomerativeClustering(n_clusters=i,affinity='euclidean', linkage='complete')
    labels = ac.fit_predict(ac_df) 
    for j in range(i):
        plt.scatter(ac_df.iloc[labels == j,0], ac_df.iloc[labels == j,1])
        plt.title('Agglomerative Clustering with {} clusters'.format(i))
        plt.plot()
    plt.show()

