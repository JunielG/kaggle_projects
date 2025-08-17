# 🤖 Machine Learning Code Examples for Freelancer Dataset

## 📊 Data Preparation

First, let's prepare your data for machine learning:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('clean_global_freelancer.csv')

# Data preprocessing
def prepare_data(df):
    # Create a copy
    data = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_country = LabelEncoder()
    le_language = LabelEncoder()
    
    data['gender_encoded'] = le_gender.fit_transform(data['gender'])
    data['country_encoded'] = le_country.fit_transform(data['country'])
    data['language_encoded'] = le_language.fit_transform(data['language'])
    
    return data, le_gender, le_country, le_language

# Prepare the dataset
data, le_gender, le_country, le_language = prepare_data(df)
print("✅ Data prepared successfully!")
print(f"Dataset shape: {data.shape}")
```

---

## 🎯 1. Rating Prediction (Classification)

### A) Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def train_random_forest_rating():
    # Features for predicting rating
    feature_cols = ['age', 'years_of_experience', 'hourly_rate', 
                   'client_satisfaction', 'gender_encoded', 'country_encoded']
    
    X = data[feature_cols]
    y = data['rating']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Results
    print("🌲 RANDOM FOREST RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Random Forest - Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return rf_model, scaler, feature_cols

# Run Random Forest
rf_model, rf_scaler, rf_features = train_random_forest_rating()
```

### B) Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm_rating():
    # Same features as Random Forest
    feature_cols = ['age', 'years_of_experience', 'hourly_rate', 
                   'client_satisfaction', 'gender_encoded', 'country_encoded']
    
    X = data[feature_cols]
    y = data['rating']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (crucial for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    # Grid search with cross-validation
    svm_grid = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("🔍 Training SVM with hyperparameter tuning...")
    svm_grid.fit(X_train_scaled, y_train)
    
    # Best model
    best_svm = svm_grid.best_estimator_
    
    # Predictions
    y_pred = best_svm.predict(X_test_scaled)
    
    # Results
    print("⚡ SVM RESULTS")
    print("=" * 50)
    print(f"Best Parameters: {svm_grid.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('SVM - Confusion Matrix')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    plt.show()
    
    return best_svm, scaler, feature_cols

# Run SVM
svm_model, svm_scaler, svm_features = train_svm_rating()
```

### C) Neural Network

```python
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def train_neural_network_rating():
    # Features for neural network
    feature_cols = ['age', 'years_of_experience', 'hourly_rate', 
                   'client_satisfaction', 'gender_encoded', 'country_encoded']
    
    X = data[feature_cols]
    y = data['rating']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (essential for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Neural Network with multiple hidden layers
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
        activation='relu',
        solver='adam',
        alpha=0.0001,           # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("🧠 Training Neural Network...")
    nn_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = nn_model.predict(X_test_scaled)
    
    # Results
    print("🧠 NEURAL NETWORK RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Number of iterations: {nn_model.n_iter_}")
    print(f"Training loss: {nn_model.loss_:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(nn_model.loss_curve_)
    plt.title('Neural Network - Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    return nn_model, scaler, feature_cols

# Run Neural Network
nn_model, nn_scaler, nn_features = train_neural_network_rating()
```

---

## 🎯 2. Market Segmentation (Clustering)

### A) K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def perform_kmeans_segmentation():
    # Features for clustering
    cluster_features = ['age', 'years_of_experience', 'hourly_rate', 
                       'rating', 'client_satisfaction']
    
    X_cluster = data[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Choose optimal k (let's use k=4)
    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_scaled)
    
    # Add cluster labels to original data
    data_clustered = data.copy()
    data_clustered['cluster'] = cluster_labels
    
    print("🎯 K-MEANS CLUSTERING RESULTS")
    print("=" * 50)
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette score: {silhouette_score(X_scaled, cluster_labels):.3f}")
    
    # Analyze clusters
    print("\n📊 Cluster Analysis:")
    for i in range(optimal_k):
        cluster_data = data_clustered[data_clustered['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data)} freelancers):")
        print(f"  Average age: {cluster_data['age'].mean():.1f}")
        print(f"  Average experience: {cluster_data['years_of_experience'].mean():.1f} years")
        print(f"  Average hourly rate: ${cluster_data['hourly_rate'].mean():.1f}")
        print(f"  Average rating: {cluster_data['rating'].mean():.1f}")
        print(f"  Average satisfaction: {cluster_data['client_satisfaction'].mean():.1f}")
        print(f"  Most common skill: {cluster_data['primary_skill'].mode().iloc[0]}")
        print(f"  Active freelancers: {cluster_data['is_active'].sum()} ({cluster_data['is_active'].mean()*100:.1f}%)")
    
    # Visualize clusters (2D projection)
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i in range(optimal_k):
        cluster_points = X_pca[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.7)
    
    # Plot centroids
    centroids_pca = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('K-Means Clustering - PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return kmeans_final, scaler, cluster_features, data_clustered

# Run K-Means
kmeans_model, cluster_scaler, cluster_features, clustered_data = perform_kmeans_segmentation()
```

### B) Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

def perform_hierarchical_clustering():
    # Features for clustering
    cluster_features = ['age', 'years_of_experience', 'hourly_rate', 
                       'rating', 'client_satisfaction']
    
    X_cluster = data[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Create dendrogram to visualize hierarchy
    # Use a sample for visualization (full dataset would be too crowded)
    sample_size = min(100, len(X_scaled))
    sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[sample_indices]
    
    plt.figure(figsize=(15, 8))
    
    # Calculate linkage matrix
    linkage_matrix = linkage(X_sample, method='ward')
    
    # Create dendrogram
    dendrogram(linkage_matrix, 
               truncate_mode='level', 
               p=5,
               leaf_rotation=90,
               leaf_font_size=10)
    
    plt.title('Hierarchical Clustering Dendrogram (Sample of 100 freelancers)')
    plt.xlabel('Freelancer Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Perform hierarchical clustering on full dataset
    n_clusters = 4
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    
    cluster_labels = hierarchical.fit_predict(X_scaled)
    
    # Add cluster labels to data
    data_hierarchical = data.copy()
    data_hierarchical['cluster'] = cluster_labels
    
    print("🌳 HIERARCHICAL CLUSTERING RESULTS")
    print("=" * 50)
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette score: {silhouette_score(X_scaled, cluster_labels):.3f}")
    
    # Analyze clusters
    print("\n📊 Cluster Analysis:")
    for i in range(n_clusters):
        cluster_data = data_hierarchical[data_hierarchical['cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data)} freelancers):")
        
        # Summary statistics
        for feature in cluster_features:
            print(f"  {feature}: {cluster_data[feature].mean():.1f} ± {cluster_data[feature].std():.1f}")
    
    # Visualize with PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i in range(n_clusters):
        cluster_points = X_pca[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.7)
    
    plt.xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Hierarchical Clustering - PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return hierarchical, scaler, cluster_features, data_hierarchical

# Run Hierarchical Clustering
hierarchical_model, hierarchical_scaler, hierarchical_features, hierarchical_data = perform_hierarchical_clustering()
```

### C) DBSCAN Clustering

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def perform_dbscan_clustering():
    # Features for clustering
    cluster_features = ['age', 'years_of_experience', 'hourly_rate', 
                       'rating', 'client_satisfaction']
    
    X_cluster = data[cluster_features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal eps using k-distance graph
    k = 4  # MinPts - 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-NN Distance')
    plt.title('K-Distance Graph for Optimal Eps')
    plt.grid(True)
    plt.show()
    
    # Choose eps based on the elbow in the k-distance graph
    eps = 0.5  # Adjust based on the plot
    min_samples = 5
    
    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Add cluster labels to data
    data_dbscan = data.copy()
    data_dbscan['cluster'] = cluster_labels
    
    # Count clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print("🎯 DBSCAN CLUSTERING RESULTS")
    print("=" * 50)
    print(f"eps: {eps}")
    print(f"min_samples: {min_samples}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 0:
        # Calculate silhouette score (excluding noise points)
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 1 and len(set(cluster_labels[non_noise_mask])) > 1:
            silhouette_avg = silhouette_score(X_scaled[non_noise_mask], cluster_labels[non_noise_mask])
            print(f"Silhouette score (excluding noise): {silhouette_avg:.3f}")
    
    # Analyze clusters
    print("\n📊 Cluster Analysis:")
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            print(f"\nNoise points ({n_noise} freelancers)")
        else:
            cluster_data = data_dbscan[data_dbscan['cluster'] == label]
            print(f"\nCluster {label} ({len(cluster_data)} freelancers):")
            
            for feature in cluster_features:
                print(f"  {feature}: {cluster_data[feature].mean():.1f} ± {cluster_data[feature].std():.1f}")
    
    # Visualize with PCA
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    
    unique_labels = set(cluster_labels)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black for noise points
            col = 'black'
            marker = 'x'
            label = 'Noise'
            alpha = 0.5
        else:
            marker = 'o'
            label = f'Cluster {k}'
            alpha = 0.7
        
        class_member_mask = (cluster_labels == k)
        xy = X_pca[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=col, marker=marker, 
                   label=label, alpha=alpha, s=50)
    
    plt.xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('DBSCAN Clustering - PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dbscan, scaler, cluster_features, data_dbscan

# Run DBSCAN
dbscan_model, dbscan_scaler, dbscan_features, dbscan_data = perform_dbscan_clustering()
```

---

## 🚀 Complete Workflow Example

```python
def complete_ml_workflow():
    print("🚀 COMPLETE ML WORKFLOW FOR FREELANCER DATASET")
    print("=" * 60)
    
    # 1. Data preparation
    print("1️⃣ Data Preparation...")
    data, le_gender, le_country, le_language = prepare_data(df)
    
    # 2. Rating prediction comparison
    print("\n2️⃣ Comparing Rating Prediction Models...")
    
    models_results = {}
    
    # Train all classification models
    rf_model, rf_scaler, rf_features = train_random_forest_rating()
    svm_model, svm_scaler, svm_features = train_svm_rating()
    nn_model, nn_scaler, nn_features = train_neural_network_rating()
    
    # 3. Market segmentation comparison
    print("\n3️⃣ Comparing Clustering Methods...")
    
    kmeans_model, cluster_scaler, cluster_features, clustered_data = perform_kmeans_segmentation()
    hierarchical_model, hierarchical_scaler, hierarchical_features, hierarchical_data = perform_hierarchical_clustering()
    dbscan_model, dbscan_scaler, dbscan_features, dbscan_data = perform_dbscan_clustering()
    
    print("\n✅ Complete workflow finished!")
    print("📊 Check the plots and results above for detailed analysis.")

# Run complete workflow
complete_ml_workflow()
```

---

## 💡 Key Takeaways

1. **Random Forest**: Best for interpretability and feature importance
2. **SVM**: Great for complex decision boundaries, needs hyperparameter tuning
3. **Neural Networks**: Can capture complex patterns but needs more data
4. **K-Means**: Good for spherical clusters, need to choose k
5. **Hierarchical**: Shows cluster hierarchy, computationally expensive
6. **DBSCAN**: Finds arbitrary shapes, handles noise well

## 🎯 Next Steps

- Try ensemble methods combining multiple models
- Feature engineering (create new features)
- Deep learning for more complex patterns
- Time series analysis if you have temporal data
- A/B testing to validate business impact