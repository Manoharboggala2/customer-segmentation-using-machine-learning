import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.impute import SimpleImputer

def preprocess_data(df, numerical_cols):
    
    """Preprocess the data by scaling numerical features"""
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
def perform_kmeans(X, n_clusters=3, random_state=42):
    """Perform K-Means clustering"""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10  # Explicitly set to suppress warning
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def perform_dbscan(X, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels, dbscan

def perform_hierarchical(X, n_clusters=3, linkage_method='ward'):
    """Perform Hierarchical clustering"""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = hierarchical.fit_predict(X)
    return labels, hierarchical

def evaluate_clustering(X, labels):
    """Evaluate clustering results"""
    if len(np.unique(labels)) < 2:
        return {
            'silhouette': -1,
            'calinski_harabasz': -1,
            'davies_bouldin': float('inf')
        }
    
    return {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    }

def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using elbow method"""
    distortions = []
    silhouette_scores = []
    K_range = range(2, max_k+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return K_range, distortions, silhouette_scores

def plot_dendrogram(X, method='ward'):
    """Plot dendrogram for hierarchical clustering"""
    linked = linkage(X, method=method)
    return linked