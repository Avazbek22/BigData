import streamlit as st
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind, pearsonr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def stochastic_gradient_descent(X, y, theta, alpha, iterations):
        m = len(y)
        cost_history = []
        for _ in range(iterations):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradient = xi.T.dot(xi.dot(theta) - yi)
                theta -= alpha * gradient
            cost = compute_cost(X, y, theta)
            if np.isinf(cost):
                break
            cost_history.append(cost)
        return theta, cost_history

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    if np.isinf(cost) or np.isnan(cost):
        return float('inf')
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        
        # Early stopping if cost explodes
        if np.isinf(cost):
            st.warning(f"Cost exploded at iteration {i} - try smaller learning rate")
            break
        cost_history.append(cost)
    return theta, cost_history

# Set page config
st.set_page_config(page_title="Formula 1 Analysis", layout="wide")

# Title
st.title("Formula 1 Race Data Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Data Overview", "Exploratory Analysis", "Statistical Tests", "Correlation Analysis", "Gradient Descent"]
)

# Load data function with caching
@st.cache_data
def load_data():
    database = r"C:\Users\avazb\Desktop\BigData\Lab1\Formula 1 Race Data\Formula1.sqlite"
    conn = sqlite3.connect(database)
    
    races = pd.read_sql_query("SELECT * FROM races", conn)
    drivers = pd.read_sql_query("SELECT * FROM drivers", conn)
    constructors = pd.read_sql_query("SELECT * FROM constructors", conn)
    results = pd.read_sql_query("SELECT * FROM results", conn)
    circuits = pd.read_sql_query("SELECT * FROM circuits", conn)
    
    conn.close()
    
    # Data processing
    constructors['constructorId'] = constructors['constructorId'].astype('int64')
    if 'url' in circuits.columns:
        circuits = circuits.drop(columns=['url'])
    
    # Merge datasets
    df = pd.merge(results, races, on='raceId', how='left', suffixes=('_results', '_races'))
    df = pd.merge(df, drivers, on='driverId', how='left', suffixes=('', '_drivers'))
    df = pd.merge(df, constructors, on='constructorId', how='left', suffixes=('', '_constructors'))
    df = pd.merge(df, circuits, on='circuitId', how='left', suffixes=('', '_circuits'))
    
    return df

df = load_data()

# Data Overview
if analysis_type == "Data Overview":
    st.header("Dataset Overview")
    
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Shape")
    rows, cols = df.shape
    st.write(f"Number of rows: {rows}")
    st.write(f"Number of columns: {cols}")
    
    st.subheader("Column Information")
    st.write(df.dtypes)

# Exploratory Analysis
elif analysis_type == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")
    
    # Numerical variables
    st.subheader("Numerical Variables Summary")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_summary = df[numeric_columns].describe().T
    st.dataframe(numeric_summary)
    
    # Categorical variables
    st.subheader("Categorical Variables Summary")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    categorical_summary = pd.DataFrame(index=categorical_columns, columns=[
        'Missing Ratio', 'Unique Values', 'Mode'
    ])
    
    for col in categorical_columns:
        categorical_summary.loc[col, 'Missing Ratio'] = df[col].isnull().mean()
        categorical_summary.loc[col, 'Unique Values'] = df[col].nunique()
        if not df[col].empty and df[col].notna().any():
            categorical_summary.loc[col, 'Mode'] = df[col].mode()[0]
    
    st.dataframe(categorical_summary)

# Statistical Tests
elif analysis_type == "Statistical Tests":
    st.header("Statistical Hypothesis Testing")
    
    # Hypothesis 1
    st.subheader("Hypothesis 1: Points Difference by Nationality")
    nationality_options = df['nationality'].unique()
    col1, col2 = st.columns(2)
    with col1:
        nat1 = st.selectbox("Select first nationality", nationality_options, index=0)
    with col2:
        nat2 = st.selectbox("Select second nationality", nationality_options, index=1)
    
    if st.button("Run Test"):
        nationality_1 = df[df['nationality'] == nat1]['points']
        nationality_2 = df[df['nationality'] == nat2]['points']
        t_stat, p_value = ttest_ind(nationality_1, nationality_2)
        
        st.write(f"T-test results between {nat1} and {nat2}:")
        st.write(f"T-statistic: {t_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.success("Statistically significant difference (p < 0.05)")
        else:
            st.warning("No statistically significant difference (p ≥ 0.05)")
    
    # Hypothesis 2
    st.subheader("Hypothesis 2: Grid vs Position Correlation")
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    df_cleaned = df.dropna(subset=['grid', 'position'])
    
    if st.button("Calculate Correlation"):
        corr, p_value = pearsonr(df_cleaned['grid'], df_cleaned['position'])
        st.write(f"Pearson correlation coefficient: {corr:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_cleaned, x='grid', y='position', alpha=0.5, ax=ax)
        ax.set_title("Grid Position vs Final Position")
        st.pyplot(fig)

# Correlation Analysis
elif analysis_type == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numerical columns available for correlation analysis")




# Gradient Descent
elif analysis_type == "Gradient Descent":
    st.header("Gradient Descent Implementation")
    
    # Data validation
    if 'grid' not in df.columns or 'points' not in df.columns:
        st.error("Required columns not found!")
        st.stop()
        
    try:
        # Data preparation
        X_raw = df['grid'].fillna(df['grid'].mean()).values
        y_raw = df['points'].values
        
        # Convert to float and check finite
        X = X_raw.astype(float)
        y = y_raw.astype(float)
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            st.error("NaN values detected after conversion!")
            st.stop()
            
        # Normalization
        X = (X - X.mean()) / X.std()
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        st.stop()
    
    # Parameters with safer defaults
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.slider("Learning rate (alpha)", 0.0001, 0.1, 0.01, 0.0001, format="%.4f")
    with col2:
        iterations = st.slider("Number of iterations", 10, 2000, 100, 10)
    
    if st.button("Run Gradient Descent"):
        with st.spinner('Running optimization...'):
            try:
                # Initialize
                theta = np.zeros(X.shape[1])
                
                # Gradient Descent
                theta_gd, cost_history_gd = gradient_descent(X, y, theta.copy(), alpha, iterations)
                
                # Stochastic GD
                theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, theta.copy(), alpha, iterations//10)  # Fewer iterations for SGD
                
                # Results
                st.success("Optimization completed!")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Cost history
                ax1.semilogy(cost_history_gd, 'b-', label='Gradient Descent')
                ax1.semilogy(cost_history_sgd, 'r--', label='Stochastic GD')
                ax1.set_title('Cost Convergence (log scale)')
                ax1.set_xlabel('Iterations')
                ax1.set_ylabel('Log(Cost)')
                ax1.legend()
                ax1.grid(True)
                
                # Predictions
                x_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
                X_pred = np.c_[np.ones(100), x_range]
                
                ax2.scatter(X[:,1], y, alpha=0.3, label='Actual Data')
                ax2.plot(x_range, X_pred.dot(theta_gd), 'b-', label='GD Prediction')
                ax2.plot(x_range, X_pred.dot(theta_sgd), 'r--', label='SGD Prediction')
                ax2.set_title('Model Predictions')
                ax2.set_xlabel('Normalized Grid Position')
                ax2.set_ylabel('Points')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                st.stop()

# Add some spacing
st.sidebar.markdown("---")
st.sidebar.info("Formula 1 Race Data Analysis Dashboard")


