from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import os

def prepare_features_for_clustering(features_df):
    """Prepare and scale user features for clustering"""
    # Select numeric features
    feature_cols = ['num_orders', 'total_items', 'num_reorders', 'reorder_rate']
    X = features_df[feature_cols].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, scaler

def find_optimal_k(X_scaled, max_k=10):
    """Find optimal number of clusters using elbow method"""
    inertias = []
    K_range = range(2, max_k+1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10,6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('results/elbow_plot.png', dpi=150)
    plt.show()
    
    # Return elbow point (heuristic)
    return 5  # Conservative choice

def fit_customer_clusters(X_scaled, n_clusters=5):
    """Fit KMeans clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/kmeans_clusters.joblib')
    print("✅ KMeans model saved to models/kmeans_clusters.joblib")
    
    return clusters

def visualize_clusters(features_df, clusters):
    import matplotlib.pyplot as plt
    import seaborn as sns

    features_df['cluster'] = clusters

    plt.figure(figsize=(15,5))

    # Plot 1
    plt.subplot(1,3,1)
    sns.boxplot(data=features_df, x='cluster', y='num_orders')
    plt.title('Num Orders by Cluster')

    # Plot 2 & 3
    for i, col in enumerate(['total_items', 'reorder_rate']):
        plt.subplot(1,3,i+2)   # ✅ FIXED HERE
        sns.boxplot(data=features_df, x='cluster', y=col)
        plt.title(f'{col} by Cluster')

    plt.tight_layout()
    plt.show()
