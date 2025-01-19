import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Set up the Streamlit app
st.title("Bias-Variance Tradeoff Visualization")
st.write("""
This app illustrates the Mean Squared Error (MSE) decomposition into Bias and Variance.
You can interactively adjust parameters to see how model complexity, dataset size, 
and noise affect Bias, Variance, and MSE.
""")

# Sidebar controls
st.sidebar.header("Simulation Settings")
noise_level = st.sidebar.slider("Noise Level", 0.1, 2.0, 0.5, step=0.1)
dataset_size = st.sidebar.slider("Dataset Size", 10, 200, 50, step=10)
model_complexity = st.sidebar.slider("Model Complexity (Polynomial Degree)", 1, 10, 3)
n_simulations = st.sidebar.slider("Number of Simulations", 10, 100, 30, step=10)

# Generate data
np.random.seed(42)
X = np.linspace(-3, 3, dataset_size).reshape(-1, 1)
true_function = np.sin(X)
y = true_function + np.random.normal(0, noise_level, size=X.shape)

# Simulate models
biases, variances, mses = [], [], []
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
true_test = np.sin(X_test)

for _ in range(n_simulations):
    y_sim = true_function + np.random.normal(0, noise_level, size=X.shape)
    model = make_pipeline(PolynomialFeatures(model_complexity), LinearRegression())
    model.fit(X, y_sim)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(true_test, predictions)
    bias = np.mean((true_test - predictions) ** 2)
    variance = np.var(predictions)
    biases.append(bias)
    variances.append(variance)
    mses.append(mse)

# Calculate averages
avg_bias = np.mean(biases)
avg_variance = np.mean(variances)
avg_mse = np.mean(mses)

# Visualization
st.subheader("Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=X_test.flatten(), y=true_test.flatten(), label="True Function", color="green")
sns.scatterplot(x=X.flatten(), y=y.flatten(), label="Training Data", color="blue")
model = make_pipeline(PolynomialFeatures(model_complexity), LinearRegression())
model.fit(X, y)
sns.lineplot(x=X_test.flatten(), y=model.predict(X_test).flatten(), label="Model Prediction", color="red")
plt.legend()
plt.title("True Function vs Model Prediction")
st.pyplot(fig)

# Bias-Variance-MSE Decomposition
st.subheader("Bias-Variance-MSE Decomposition")
st.write(f"**Average Bias:** {avg_bias:.4f}")
st.write(f"**Average Variance:** {avg_variance:.4f}")
st.write(f"**Average MSE:** {avg_mse:.4f}")

# Bar plot of decomposition
fig2, ax2 = plt.subplots(figsize=(6, 4))
components = ['Bias', 'Variance', 'Noise']
values = [avg_bias, avg_variance, avg_mse - avg_bias - avg_variance]
sns.barplot(x=components, y=values, palette="viridis")
plt.title("MSE Decomposition")
plt.ylabel("Value")
plt.ylim(-1, 1)
st.pyplot(fig2)
