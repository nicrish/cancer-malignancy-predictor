#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 22:39:41 2026

@author: nicolerishwain
"""

####-----------------------------------------------------------
#### PCA and Pairwise Correlation
####------------------------------------------------------------

## Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from IPython.display import clear_output
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import fcluster

##This is a small sample of the Iris dataset (link above)
## !!! This is MY path :) YOU need to update this to be YOUR path !!!
path = "/Users/nicolerishwain/src/CS432/ML/breast_cancer_clean.csv"
DF = pd.read_csv(path, index_col=False)

print(DF)

##--------------------------------
## Remove and save the label
## Next, update the label so that 
## rather than names like "Iris-setosa"
## we use numbers instead. 
## This will be necessary when we "color"
## the data in our plot
##---------------------------------------
DFLabel = DF["diagnosis"]  ## Save the Label 
print(DFLabel)  ## print the labels
print(type(DFLabel))  ## check the datatype you have

## Remap the label names from strings to numbers
MyDic = {"M": 0, "B": 1}
DFLabel = DFLabel.map(MyDic)  ## Update the label to your number remap values
print(DFLabel) ## Print the labels to confirm 

## Now, remove the label from the original dataframe
DF = DF.drop(["diagnosis"], axis=1)
print(DF) #Print the dataframe to confirm 
DF = DF.drop(["Unnamed: 0","compactness_mean","concavity_mean","concave points_mean",
              "symmetry_mean","fractal_dimension_mean","radius_se",
              "texture_se","perimeter_se","area_se","smoothness_se",	
              "compactness_se",	"concavity_se","concave points_se",
              "symmetry_se"	,"fractal_dimension_se","radius_worst",	
              "texture_worst","perimeter_worst","area_worst",
              "smoothness_worst","compactness_worst","concavity_worst",
              "concave points_worst","symmetry_worst","fractal_dimension_worst"], axis=1)
print(DF.dtypes)
###-------------------------------------------
### kmeans
###-------------------------------------------

# Scale data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(DF)

# Compute WSS
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Plot elbow
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WSS / SSE")
plt.title("Elbow Method for KMeans")
plt.show()

###---------------------------
### Run KMeans Clustering 2
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=2, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 2")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_

print(Centroids)

DF['cluster'] = PredictedClusters


sns.pairplot(DF, hue='cluster', diag_kind='kde', palette='viridis')
plt.show()

###---------------------------
### Run KMeans Clustering 3
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=3, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 3")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_
print(Centroids)

DF['cluster'] = PredictedClusters


sns.pairplot(DF, hue='cluster', diag_kind='kde', palette='viridis')
plt.show()

###---------------------------
### Run KMeans Clustering 4
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=4, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 4")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_
print(Centroids)

DF['cluster'] = PredictedClusters


sns.pairplot(DF, hue='cluster', diag_kind='kde', palette='viridis')
plt.show()

###-------------------------------------------
### Standardize your dataset
###-------------------------------------------
DF = DF.drop(["cluster"], axis=1)
print(DF)
scaler = StandardScaler() ##Instantiate
DF = scaler.fit_transform(DF) ## Scale data
print("xxxxxxx")
print(DF)

###############################################
###--------------PERFORM PCA------------------
###############################################
## Instantiate PCA and choose how many components
DF_PCA = DF.copy()
MyPCA = PCA(n_components=3)
Result = MyPCA.fit_transform(DF_PCA)
## Print the values of the first component 
print("xxxxxxx")
print(Result[:, 0]) 
print("xxxxxxx")
print(Result) ## Print the new (transformed) dataset
print("The relative eigenvalues are:", MyPCA.explained_variance_ratio_)
print("The actual eigenvalues are:", MyPCA.explained_variance_)
EVects = MyPCA.components_
print("The eigenvectors are:\n", EVects)

#################################################
## Visualize the transformed 3D dataset
## we just created using PCA
#################################################
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')

x = Result[:, 0]
y = Result[:, 1] 
z = Result[:, 2]

scatter = ax2.scatter(
    x, y, z,
    c=DFLabel,
    cmap="RdYlGn",
    edgecolor='k',
    s=200
)

# ax2.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=200, c=DFLabel)
plt.colorbar(scatter, label='diagnosis')
ax2.set_xlabel('PC0')
ax2.set_ylabel('PC1')
ax2.set_zlabel('PC2')
ax2.set_title('3D PCA')

plt.show()


############################################
## Create Plot to Show Eigenvalues
############################################
plt.bar(
    range(len(MyPCA.explained_variance_ratio_)), 
    MyPCA.explained_variance_ratio_,
    alpha=0.5, align='center', label='Individual Explained Variances'
)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title("Eigenvalues: Percentage of Variance/Information")
plt.tight_layout()
plt.show()

###############################################
## Create a DF of the most important features
##################################################
shape = MyPCA.components_.shape[0]
#print(shape)
feature_names = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"]

most_important = [np.abs(MyPCA.components_[i]).argmax() for i in range(shape)]
most_important_names = [feature_names[most_important[i]] for i in range(shape)]

## Build a dictionary of the important features by PC
MyDic = {'PC{}'.format(i): most_important_names[i] for i in range(shape)}

## Build the dataframe
Important_DF = pd.DataFrame(MyDic.items())
print("xxxxxxx")
print(Important_DF)

###-------------------------------------------
### kmeans
###-------------------------------------------

# Scale data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(DF_PCA)

# Compute WSS
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Plot elbow
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WSS / SSE")
plt.title("Elbow Method for KMeans")
plt.show()

###---------------------------
### Run KMeans Clustering 2
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=2, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 2")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_

print(Centroids)

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')

x = Result[:, 0]
y = Result[:, 1] 
z = Result[:, 2]

scatter = ax2.scatter(
    x, y, z,
    c=PredictedClusters,
    cmap="RdYlGn",
    edgecolor='k',
    s=200
)

# ax2.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=200, c=DFLabel)
plt.colorbar(scatter, label='kmeans clusters')
ax2.set_xlabel('PC0')
ax2.set_ylabel('PC1')
ax2.set_zlabel('PC2')
ax2.set_title('3D PCA with 2 clusters')

plt.show()

###---------------------------
### Run KMeans Clustering 3
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=3, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 3")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_
print(Centroids)

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')

x = Result[:, 0]
y = Result[:, 1] 
z = Result[:, 2]

scatter = ax2.scatter(
    x, y, z,
    c=PredictedClusters,
    cmap="RdYlGn",
    edgecolor='k',
    s=200
)

# ax2.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=200, c=DFLabel)
plt.colorbar(scatter, label='kmeans clusters')
ax2.set_xlabel('PC0')
ax2.set_ylabel('PC1')
ax2.set_zlabel('PC2')
ax2.set_title('3D PCA with 3 clusters')

plt.show()
###---------------------------
### Run KMeans Clustering 4
###---------------------------

## Instantiation
KMeansInit = KMeans(n_clusters=4, max_iter=200)
MyKMeans=KMeansInit.fit(DF)
print("kmeans 4")
# Get cluster assignment labels
PredictedClusters =MyKMeans.labels_
print(PredictedClusters)



## See the centroids
Centroids=MyKMeans.cluster_centers_
print(Centroids)

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')

x = Result[:, 0]
y = Result[:, 1] 
z = Result[:, 2]

scatter = ax2.scatter(
    x, y, z,
    c=PredictedClusters,
    cmap="RdYlGn",
    edgecolor='k',
    s=200
)

# ax2.scatter(x, y, z, cmap="RdYlGn", edgecolor='k', s=200, c=DFLabel)
plt.colorbar(scatter, label='kmeans clusters')
ax2.set_xlabel('PC0')
ax2.set_ylabel('PC1')
ax2.set_zlabel('PC2')
ax2.set_title('3D PCA with 4 clusters')

plt.show()


###---------------------------------------------------
### Run Hierarchical Clustering (Agglomerative)
###----------------------------------------------------

## Instantiation
MyAggClusterInit = AgglomerativeClustering(n_clusters = 3) 

## Reference:
## https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

## Perform the clustering
MyAggClusters=MyAggClusterInit.fit(DF)
PredictedClustersAgg=MyAggClusters.labels_
print(PredictedClustersAgg)

###-------------------
### Plot a Dendrogram
###-------------------

### Note: CLOSE any other plots from above before running this code below

### Tutorial: 
## https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

## To plot a dendrogram, you
## need to create a matrix first
Z = linkage(DF)
#dendrogram(Z)
plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True)
plt.show()



###---------------------------------------------------
### Run Hierarchical Clustering (Agglomerative)
###----------------------------------------------------


"""
Truncated Dendrogram Visualization Examples
This script shows different ways to truncate dendrograms for better interpretation
"""



# Generate sample data (replace this with your own data)
# Creating 500 samples with 5 clusters for demonstration
X = DF.copy()

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Original full dendrogram (not recommended for many points)
ax1 = axes[0, 0]
dendrogram(Z, ax=ax1)
ax1.set_title('Original Dendrogram (Hard to Read)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Distance')

# 2. Truncated dendrogram - showing only last p merges
ax2 = axes[0, 1]
dendrogram(Z, truncate_mode='lastp', p=30, ax=ax2)
ax2.set_title('Truncated: Last 30 Merges Only', fontsize=14, fontweight='bold')
ax2.set_xlabel('Cluster Size')
ax2.set_ylabel('Distance')
ax2.text(0.02, 0.98, 'Shows only the top 30 cluster merges\nNumbers in parentheses = cluster size',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Truncated dendrogram - showing level
ax3 = axes[1, 0]
dendrogram(Z, truncate_mode='level', p=5, ax=ax3)
ax3.set_title('Truncated: Level 5', fontsize=14, fontweight='bold')
ax3.set_xlabel('Cluster Size')
ax3.set_ylabel('Distance')
ax3.text(0.02, 0.98, 'Shows clustering hierarchy up to level 5\nGood for seeing hierarchical structure',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 4. Truncated with color threshold
ax4 = axes[1, 1]
# Calculate a good color threshold (e.g., 70% of max distance)
max_d = np.max(Z[:, 2])
color_threshold = 0.7 * max_d
dendrogram(Z, truncate_mode='lastp', p=20, ax=ax4, 
           color_threshold=color_threshold,
           above_threshold_color='gray')
ax4.set_title(f'Truncated with Color Threshold (distance={color_threshold:.2f})', 
              fontsize=14, fontweight='bold')
ax4.set_xlabel('Cluster Size')
ax4.set_ylabel('Distance')
ax4.axhline(y=color_threshold, c='red', linestyle='--', linewidth=2, label='Color threshold')
ax4.legend()
ax4.text(0.02, 0.98, 'Different colors = different clusters\nClusters above red line are colored gray',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()


# Create a detailed example with annotations
fig, ax = plt.subplots(figsize=(14, 8))
dend = dendrogram(Z, truncate_mode='lastp', p=15, ax=ax,
                  color_threshold=color_threshold,
                  above_threshold_color='#808080')
ax.set_title('Truncated Dendrogram ', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Cluster Index (number in parentheses = size)', fontsize=12)
ax.set_ylabel('Merge Distance (Higher = More Dissimilar)', fontsize=12)
ax.axhline(y=color_threshold, c='red', linestyle='--', linewidth=2, 
           label=f'Suggested cut height: {color_threshold:.2f}')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# # Add interpretation guide
# guide_text = """
# HOW TO INTERPRET:
# • Height of merge: How different the clusters being joined are
# • Horizontal line length: Indicates cluster quality (longer = more distinct)
# • Numbers in parentheses: Number of original data points in that cluster
# • Colors: Different clusters when cut at the red threshold
# • To choose clusters: Draw a horizontal line and count vertical lines it crosses
# """
ax.text(1.02, 0.5, guide_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=dict(boxstyle='round', 
        facecolor='yellow', alpha=0.3))

plt.tight_layout()


# Example: How to use truncation with your actual data
print("\n" + "="*60)
print("HOW TO USE WITH YOUR DATA:")
print("="*60)
print("""
# 1. Load your data
# X = your_data  # shape: (n_samples, n_features)

# 2. Perform hierarchical clustering
# Z = linkage(X, method='ward')  # or 'complete', 'average', etc.

# 3. Create truncated dendrogram
# plt.figure(figsize=(12, 6))
# dendrogram(Z, truncate_mode='lastp', p=30)
# plt.title('Truncated Dendrogram')
# plt.xlabel('Cluster Size')
# plt.ylabel('Distance')
# plt.show()

Key parameters to adjust:
- truncate_mode='lastp': Shows only last p merges (RECOMMENDED)
- truncate_mode='level': Shows hierarchy up to level p
- p: Number to show (try 15-40 for lastp, 3-7 for level)
- color_threshold: Height to color different clusters
""")

print("\n✓ Script completed! Check the output images.")
