# importing libraries    
import numpy as nm    
import matplotlib.pyplot as mtp    
import pandas as pd    

# ✅ Correct path to the CSV file
dataset = pd.read_csv('C:/Users/Icon/Downloads/ML--LAB-main/ML--LAB-main/ML6/Mall_Customers.csv')  

# ✅ Using Annual Income and Spending Score for clustering
x = dataset.iloc[:, [3, 4]].values 

# Importing KMeans
from sklearn.cluster import KMeans  

# List to store WCSS values
wcss_list = []  

# ✅ Elbow Method to determine optimal clusters
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  
    kmeans.fit(x)  
    wcss_list.append(kmeans.inertia_)  

# ✅ Plotting the Elbow Graph
mtp.plot(range(1, 11), wcss_list)  
mtp.title('The Elbow Method Graph')  
mtp.xlabel('Number of Clusters (k)')  
mtp.ylabel('WCSS')  
mtp.show() 

# ✅ KMeans with optimal k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)  
y_predict = kmeans.fit_predict(x) 

# ✅ Visualizing the clusters
mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s=100, c='blue', label='Cluster 1')  
mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s=100, c='green', label='Cluster 2')  
mtp.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s=100, c='red', label='Cluster 3')  
mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s=100, c='cyan', label='Cluster 4')  
mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s=100, c='magenta', label='Cluster 5')  

# ✅ Plotting centroids
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', label='Centroids')   

mtp.title('Clusters of Customers')  
mtp.xlabel('Annual Income (k$)')  
mtp.ylabel('Spending Score (1-100)')  
mtp.legend()  
mtp.show()
