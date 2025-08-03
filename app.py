import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from clustering_utils import *
from sklearn.impute import SimpleImputer

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# Title
st.title("Customer Segmentation Dashboard")
st.markdown("""
This dashboard allows you to perform customer segmentation using different clustering algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering
""")

# Sidebar for upload and parameters
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    st.header("Clustering Parameters")
    algorithm = st.selectbox(
        "Select clustering algorithm",
        ["K-Means", "DBSCAN", "Hierarchical"]
    )
    
    if algorithm == "K-Means":
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        random_state = st.slider("Random state", 0, 100, 42)
    
    elif algorithm == "DBSCAN":
        eps = st.slider("EPS (neighborhood radius)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Minimum samples", 1, 20, 5)
    
    elif algorithm == "Hierarchical":
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        linkage_method = st.selectbox(
            "Linkage method",
            ["ward", "complete", "average", "single"]
        )
    
    st.header("Visualization Settings")
    x_axis = st.selectbox("X-axis feature", [])
    y_axis = st.selectbox("Y-axis feature", [])
# Replace this:
if uploaded_file is None:
    df = pd.read_csv("file.csv")
else:
    df = pd.read_csv("file.csv")

# With this:
if uploaded_file is None:
    st.warning("Using sample data. Upload your own CSV file to analyze your data.")
    df = pd.DataFrame({
        'Age': np.random.normal(35, 10, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'SpendingScore': np.random.randint(1, 100, 100)
    })
else:
    df = pd.read_csv(uploaded_file)

# # Load sample data if no file uploaded
# if uploaded_file is None:
#     st.warning("Using sample data. Upload your own CSV file to analyze your data.")
#     df = pd.read_csv("file.csv")
# else:
#     df = pd.read_csv("file.csv")
# Handle missing values right after loading
if df.isna().any().any():
    st.warning("Data contains missing values. Imputing with mean values.")
    imputer = SimpleImputer(strategy='mean')
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols:
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Display raw data
with st.expander("Show Raw Data"):
    st.dataframe(df)

# Select numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numerical_cols:
    st.error("No numerical columns found in the data. Please upload a dataset with numerical features.")
    st.stop()

# Preprocess data
df_processed = preprocess_data(df.copy(), numerical_cols)
X = df_processed[numerical_cols].values

# Update visualization options
with st.sidebar:
    st.header("Visualization Settings")
    x_axis = st.selectbox("X-axis feature", numerical_cols, index=0)
    y_axis = st.selectbox("Y-axis feature", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)

# Perform clustering
if algorithm == "K-Means":
    labels, model = perform_kmeans(X, n_clusters=n_clusters, random_state=random_state)
    st.session_state['current_model'] = model
elif algorithm == "DBSCAN":
    labels, model = perform_dbscan(X, eps=eps, min_samples=min_samples)
    st.session_state['current_model'] = model
elif algorithm == "Hierarchical":
    labels, model = perform_hierarchical(X, n_clusters=n_clusters, linkage_method=linkage_method)
    st.session_state['current_model'] = model

# Add labels to dataframe
df['Cluster'] = labels

# Evaluation metrics
metrics = evaluate_clustering(X, labels)

# Layout for results
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cluster Distribution")
    fig = px.pie(df, names='Cluster', title='Cluster Distribution')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Evaluation Metrics")
    metric_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index'],
        'Value': [metrics['silhouette'], metrics['calinski_harabasz'], metrics['davies_bouldin']]
    })
    st.dataframe(metric_df.style.format({'Value': '{:.3f}'}), hide_index=True)

# Scatter plot
st.subheader("Cluster Visualization")
fig = px.scatter(
    df, 
    x=x_axis, 
    y=y_axis, 
    color='Cluster',
    title=f'{algorithm} Clustering',
    hover_data=df.columns
)
st.plotly_chart(fig, use_container_width=True)

# Feature distributions by cluster
st.subheader("Feature Distributions by Cluster")
selected_features = st.multiselect(
    "Select features to visualize", 
    numerical_cols, 
    default=numerical_cols[:2] if len(numerical_cols) >= 2 else numerical_cols
)

if selected_features:
    for feature in selected_features:
        fig = px.box(df, x='Cluster', y=feature, title=f'{feature} Distribution by Cluster')
        st.plotly_chart(fig, use_container_width=True)

# Compare algorithms
st.header("Algorithm Comparison")

if st.button("Compare All Algorithms"):
    with st.spinner("Running all algorithms for comparison..."):
        # K-Means
        kmeans_labels, _ = perform_kmeans(X, n_clusters=4)
        kmeans_metrics = evaluate_clustering(X, kmeans_labels)
        
        # DBSCAN
        dbscan_labels, _ = perform_dbscan(X, eps=0.5, min_samples=5)
        dbscan_metrics = evaluate_clustering(X, dbscan_labels)
        
        # Hierarchical
        hierarchical_labels, _ = perform_hierarchical(X, n_clusters=4)
        hierarchical_metrics = evaluate_clustering(X, hierarchical_labels)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
            'Silhouette': [kmeans_metrics['silhouette'], dbscan_metrics['silhouette'], hierarchical_metrics['silhouette']],
            'Calinski-Harabasz': [kmeans_metrics['calinski_harabasz'], dbscan_metrics['calinski_harabasz'], hierarchical_metrics['calinski_harabasz']],
            'Davies-Bouldin': [kmeans_metrics['davies_bouldin'], dbscan_metrics['davies_bouldin'], hierarchical_metrics['davies_bouldin']],
            'Clusters Found': [len(np.unique(kmeans_labels)), len(np.unique(dbscan_labels[dbscan_labels != -1])), len(np.unique(hierarchical_labels))]
        })
        
        st.dataframe(comparison_df.style.format({
            'Silhouette': '{:.3f}',
            'Calinski-Harabasz': '{:.3f}',
            'Davies-Bouldin': '{:.3f}'
        }), hide_index=True)
        
        # Visual comparison
        fig = px.bar(
            comparison_df.melt(id_vars='Algorithm', value_vars=['Silhouette', 'Calinski-Harabasz']), 
            x='Algorithm', 
            y='value', 
            color='variable',
            barmode='group',
            title='Algorithm Comparison (Higher is better)',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.bar(
            comparison_df.melt(id_vars='Algorithm', value_vars=['Davies-Bouldin']), 
            x='Algorithm', 
            y='value', 
            color='variable',
            title='Algorithm Comparison (Lower is better for Davies-Bouldin)',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)

# K-Means specific analysis
if algorithm == "K-Means":
    st.header("K-Means Analysis")
    
    st.subheader("Elbow Method")
    K_range, distortions, silhouette_scores = find_optimal_k(X, max_k=10)
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion (Inertia)', color=color)
    ax1.plot(K_range, distortions, 'bo-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(K_range, silhouette_scores, 'bo-', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Elbow Method and Silhouette Score for Optimal K')
    st.pyplot(fig)

# Hierarchical specific analysis
if algorithm == "Hierarchical":
    st.header("Hierarchical Clustering Analysis")
    
    st.subheader("Dendrogram")
    linked = plot_dendrogram(X, method=linkage_method)
    
    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        truncate_mode='lastp',
        p=30
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    st.pyplot(plt)

# Download results
st.header("Download Results")
if st.button("Download Cluster Data"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )