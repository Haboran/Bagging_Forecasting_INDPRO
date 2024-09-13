# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:51:56 2024

@author: Haboran
"""
#%%
#clear the console and delete all the variables

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
get_ipython().run_line_magic('clear', '/')
 
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import Ridge, Lasso
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import warnings


#%%
# Transformation function definition
def transform_function(series, tcode):
    if tcode == 1:
        return series
    elif tcode == 2:
        return series.diff()
    elif tcode == 3:
        return series.diff().diff()
    elif tcode == 4:
        return np.log(series)
    elif tcode == 5:
        return np.log(series).diff()
    elif tcode == 6:
        return np.log(series).diff().diff()
    else:
        return (series / series.shift(1) - 1) - (series.shift(1) / series.shift(2) - 1)

def block_bootstrap(data, block_size):
    n = len(data)
    num_blocks = n // block_size + 1
    blocks = [data.iloc[i:i + block_size] for i in range(0, n, block_size)]
    bootstrap_sample = []

    while len(bootstrap_sample) < n:
        block = blocks[np.random.randint(0, len(blocks))]
        bootstrap_sample.extend(block.values.tolist())

    bootstrap_sample = bootstrap_sample[:n]  # Trim to match the original length
    return pd.DataFrame(bootstrap_sample, columns=data.columns, index=data.index)

# AIC-based lag selection for the AR model
def select_lag(train_data, max_lags=12):
    best_aic = np.inf
    best_lag = 0
    for lag in range(1, max_lags + 1):
        model = AutoReg(train_data, lags=lag).fit()
        aic = model.aic
        if aic < best_aic:
            best_aic = aic
            best_lag = lag
    return best_lag

#%%
# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load and preprocess the dataset
data_path = 'Fred-2023-12.csv'
data = pd.read_csv(data_path)

# Extract transformation codes and drop the first row
tcodes = data.iloc[0, 1:]
data = data.iloc[1:]

# Set the index to 'sasdate' and parse dates
data.set_index('sasdate', inplace=True)
data.index = pd.to_datetime(data.index, format='%m/1/%Y')

# Apply transformations based on tcodes
for col in data.columns:
    data[col] = transform_function(data[col].astype(float), int(tcodes[col]))

# Number of lags to include (adjust this as needed)
lags = 12

columns_to_lag = data.columns  # All columns except index

# Create lagged columns for all variables except 'sasdate'
for column in columns_to_lag:
    for i in range(1, lags + 1):
        data[f'{column}_Lag{i}'] = data[column].shift(i)

# # Drop rows where lagged values are missing (optional, as this will remove the rows at the beginning)
# data_lagged = data.dropna(subset=[f'INDPRO_Lag{i}' for i in range(1, lags + 1)])


# Select data starting from 1992
data = data['1982-01-01':]

# Drop rows with any NaN values after transformations
data = data.dropna(axis=1)

# Plotting 'INDPRO'
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['INDPRO'], label='INDPRO', color='blue')
plt.title('Industrial Production Index (INDPRO) Over Time')
plt.xlabel('Year')
plt.ylabel('INDPRO Value')
plt.legend()
plt.grid(True)
plt.show()

# # Calculate and print correlation of all variables with 'INDPRO'
# correlations_initial = data.corrwith(data['INDPRO'])
# print("Correlations of all variables with INDPRO:")
# print(correlations_initial)
end_train = '2022-07-01'
start_test = '2021-06-01'
#end_test = '2023-12-01'
end_test = '2023-11-01'
# Split data into training and testing sets for rolling forecast
train_rolling = data['1982-01-01':'2021-06-01']
#test_rolling = data['2018-01-01':'2023-12-01']
test_rolling = data['2021-06-01':'2023-11-01']

# Ensure both datasets include the same features
common_columns_rolling = train_rolling.columns.intersection(test_rolling.columns)
train_rolling = train_rolling[common_columns_rolling]
test_rolling = test_rolling[common_columns_rolling]
    
# Rolling forecast process with block bootstrap
# Lambdas (alpha values) for Ridge and Lasso
ridge_alphas = [0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 200]
lasso_alphas = [1, 2, 3, 4, 5, 10, 20, 50, 100]

# Initialize tracking structures for rolling forecast
feature_usage_count_rolling = {}
model_config_count_rolling = {}
mse_benchmark_all = []
best_predicts_ridge = []
best_predicts_lasso = []
best_mse_ridge_rolling = []
best_mse_lasso_rolling = []
best_alphas_ridge = []
best_alphas_lasso = []
mse_scores_rolling = []
predictions_rolling = []
true_values_rolling = []
predictions_benchmark = []
# Initialize a DataFrame to store MSEs for each feature count
mse_feature_count = pd.DataFrame(columns=range(1, 21))  # Columns for feature count from 1 to 20

# block_size = 12  # Define block size for block bootstrap

# Prepare training data with block bootstrap
# train_rolling_bootstrap = block_bootstrap(train_rolling, block_size)

y_train_rolling = train_rolling['INDPRO'].shift(1).iloc[1:]  # Exclude the last period
X_train_rolling = train_rolling.drop('INDPRO', axis=1).iloc[:-1]  # Shift by one period and exclude the first period

# Select the best lag based on AIC
best_lag = select_lag(y_train_rolling)

# Fit the benchmark model using the selected lag
benchmark_model = AutoReg(y_train_rolling, lags=best_lag).fit()

# Rolling forecast process with block bootstrap
for t in range(len(test_rolling) - 1):
    
   # y_test_rolling = test_rolling.iloc[t + 1]['INDPRO']  # Predicting y_t+1
    y_test_rolling = test_rolling.iloc[[t + 1]]['INDPRO']
    X_test_rolling = test_rolling.iloc[t:t + 1].drop('INDPRO', axis=1)
    
    # Get the index of the last training data point
    last_train_index = y_train_rolling.index[-1]

    # Forecast the next value using the benchmark model
    forecast = benchmark_model.predict(start=last_train_index, end=last_train_index)[0]
    
    predictions_benchmark.append(forecast)
    mse_benchmark = mean_squared_error([y_test_rolling], [forecast])
    
    mse_benchmark_all.append(mse_benchmark)
    # Update the model with the new data point
    benchmark_model = AutoReg(y_train_rolling, lags=best_lag).fit()
    
    # Track best MSE and corresponding alpha for Ridge and Lasso
    best_ridge_mse = float('inf')
    best_lasso_mse = float('inf')
    best_ridge_alpha = None
    best_lasso_alpha = None

    # Ridge Regression: Loop through all ridge_alphas
    for alpha in ridge_alphas:
        ridge_model_rolling = Ridge(alpha=alpha)
        ridge_model_rolling.fit(X_train_rolling, y_train_rolling)
        
        # Prediction and MSE
        ridge_prediction_rolling = ridge_model_rolling.predict(X_test_rolling)[0]
        ridge_mse_current_rolling = mean_squared_error([y_test_rolling], [ridge_prediction_rolling])
        
        if ridge_mse_current_rolling < best_ridge_mse:
            best_ridge_mse = ridge_mse_current_rolling
            best_ridge_alpha = alpha
            best_ridge_predict = ridge_prediction_rolling
    # Lasso Regression: Loop through all lasso_alphas
    for alpha in lasso_alphas:
        lasso_model_rolling = Lasso(alpha=alpha)
        lasso_model_rolling.fit(X_train_rolling, y_train_rolling)

        # Prediction and MSE
        lasso_prediction_rolling = lasso_model_rolling.predict(X_test_rolling)[0]
        lasso_mse_current_rolling = mean_squared_error([y_test_rolling], [lasso_prediction_rolling])
        
        if lasso_mse_current_rolling < best_lasso_mse:
            best_lasso_mse = lasso_mse_current_rolling
            best_lasso_alpha = alpha
            best_lasso_predict = lasso_prediction_rolling
            
    # Store best results for Ridge and Lasso
    best_mse_ridge_rolling.append(best_ridge_mse)
    best_alphas_ridge.append(best_ridge_alpha)
    best_predicts_ridge.append(best_ridge_predict)
    
    best_mse_lasso_rolling.append(best_lasso_mse)
    best_alphas_lasso.append(best_lasso_alpha)
    best_predicts_lasso.append(best_lasso_predict)

    # # Optionally, average the predictions from the best Ridge and Lasso models
    # final_prediction_rolling = (ridge_prediction_rolling + lasso_prediction_rolling) / 2
    # predictions_rolling.append(final_prediction_rolling)
    # true_values_rolling.append(y_test_rolling)
    
    # Calculate the absolute correlation with 'INDPRO'
    # Ensure the indices of X_train_rolling and y_train_rolling match
    # X_train_rolling, y_train_rolling = X_train_rolling.align(y_train_rolling, axis=0, join='inner')
    
    # Calculate the absolute correlation with 'INDPRO'
    correlations_rolling = X_train_rolling.corrwith(y_train_rolling).abs()

    # Sort and take the top 20 most correlated features
    top_features_rolling = correlations_rolling.sort_values(ascending=False).head(40).index

    # Update feature usage count
    for feature in top_features_rolling:
        feature_usage_count_rolling[feature] = feature_usage_count_rolling.get(feature, 0) + 1

    # Update model configuration usage
    model_config_rolling = tuple(sorted(top_features_rolling))
    model_config_count_rolling[model_config_rolling] = model_config_count_rolling.get(model_config_rolling, 0) + 1

    # Restrict X_train and X_test to these top features
    X_train_rolling_bagging = X_train_rolling[top_features_rolling]
    X_test_rolling_bagging = X_test_rolling[top_features_rolling]

    # Initialize and fit the Bagging Regressor
    bagging_model_rolling = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=500, random_state=42)
    bagging_model_rolling.fit(X_train_rolling_bagging, y_train_rolling)

    # Forecast for the current period
    prediction_rolling = bagging_model_rolling.predict(X_test_rolling_bagging)[0]

    # Store predictions and true values
    predictions_rolling.append(prediction_rolling)
    true_values_rolling.append(y_test_rolling)

    # Calculate MSE for the current forecast
    mse_current_rolling = mean_squared_error([y_test_rolling], [prediction_rolling])
    mse_scores_rolling.append(mse_current_rolling)
    
    # DataFrame to store MSE for different feature counts in this iteration
    mse_for_t = {}

    for feature_count in range(1, 21):  # Loop through feature counts from 1 to 20

        # Calculate the absolute correlation with 'INDPRO'
        correlations_rolling = X_train_rolling.corrwith(y_train_rolling).abs()

        # Sort and take the top N most correlated features (N = feature_count)
        top_features_rolling = correlations_rolling.sort_values(ascending=False).head(feature_count).index

        # Update feature usage count
        for feature in top_features_rolling:
            feature_usage_count_rolling[feature] = feature_usage_count_rolling.get(feature, 0) + 1

        # Update model configuration usage
        model_config_rolling = tuple(sorted(top_features_rolling))
        model_config_count_rolling[model_config_rolling] = model_config_count_rolling.get(model_config_rolling, 0) + 1

        # Restrict X_train and X_test to these top features
        X_train_rolling_bagging = X_train_rolling[top_features_rolling]
        X_test_rolling_bagging = X_test_rolling[top_features_rolling]

        # Initialize and fit the Bagging Regressor
        bagging_model_rolling = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=500, random_state=42)
        bagging_model_rolling.fit(X_train_rolling_bagging, y_train_rolling)

        # Forecast for the current period
        prediction_rolling = bagging_model_rolling.predict(X_test_rolling_bagging)[0]

        # Calculate MSE for the current forecast
        mse_current_rolling = mean_squared_error([y_test_rolling.iloc[0]], [prediction_rolling])

        # Store MSE for the current feature count
        mse_for_t[feature_count] = mse_current_rolling

    # Append the MSEs for this iteration (t) to the DataFrame
    mse_feature_count = pd.concat([mse_feature_count, pd.DataFrame([mse_for_t])])
    
    #Append actual test observation to training data for next iteration
    new_row_rolling = test_rolling.iloc[t:t+1]
    train_rolling = pd.concat([train_rolling, new_row_rolling])

    # Append the new test observation for X_train and y_train for the next iteration
    X_train_rolling = pd.concat([X_train_rolling, X_test_rolling])
    y_train_rolling = pd.concat([y_train_rolling, y_test_rolling])  # Fixed: No need for index


# Print results
print("Feature Usage Count:")
for feature, count in feature_usage_count_rolling.items():
    print(f"{feature}: {count}")

print("Model Configurations, Their Counts, and Number of Variables Used:")
for config, count in model_config_count_rolling.items():
    num_variables = len(config)  # Count the number of variables in the tuple
    print(f"Configuration: {config}, Count: {count}, Variables used: {num_variables}")

print("\nMSE Scores for Each Prediction Period:")
for i, mse in enumerate(mse_scores_rolling):
    print(f"Period {i+1}: {mse}")

mean_MSE = sum(mse_benchmark_all) / len(mse_benchmark_all)
print(f"The mean of all predicted MSE for the benchmark model is: {mean_MSE}")
mean_MSE = sum(mse_scores_rolling) / len(mse_scores_rolling)
print(f"The mean of all predicted MSE for bagging is: {mean_MSE}")
mean_MSE_ridge = sum(best_mse_ridge_rolling) / len(best_mse_ridge_rolling)
print(f"The mean of all predicted MSE for ridge is: {mean_MSE_ridge}")
mean_MSE_lasso = sum(best_mse_lasso_rolling) / len(best_mse_lasso_rolling)
print(f"The mean of all predicted MSE for lasso is: {mean_MSE_lasso}")

mean_mse_per_feature_count = mse_feature_count.mean()

# Create a new DataFrame to store the results
mean_mse_table = pd.DataFrame({
    'Amount of Features': mean_mse_per_feature_count.index,
    'Mean MSE': mean_mse_per_feature_count.values
})

# Print the table
print(mean_mse_table)

test_rolling = test_rolling[1:]
# Plotting the true values and predictions
# Plot historical data from 2005 to 2014
plt.figure(figsize=(12, 6))
plt.plot(data['2021-01-01':'2023-11-01'].index, data['2021-01-01':'2023-11-01']['INDPRO'], label='True Values (2021-2023)', color='blue')

# Plot true values and predictions from 2015 onwards
plt.plot(test_rolling.index, predictions_benchmark, label='Predictions Benchmark(from 06.2021)', color='purple', linestyle='--')
plt.plot(test_rolling.index, predictions_rolling, label='Predictions Bagging(from 06.2021)', color='red', linestyle='--')
plt.plot(test_rolling.index, best_predicts_ridge, label='Predictions RIDGE(from 06.2021)', color='green', linestyle='--')
plt.plot(test_rolling.index, best_predicts_lasso, label='Predictions LASSO(from 06.2021)', color='orange', linestyle='--')

plt.title('INDPRO: Historical Data and Forecast Comparison for a forecast horizon(y_t+1)')
plt.xlabel('Year')
plt.ylabel('INDPRO Value')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Split data into training and testing sets for rolling forecast
train_rolling = data['1982-01-01':end_train]
test_rolling = data[start_test:end_test]

# Ensure both datasets include the same features
common_columns_rolling = train_rolling.columns.intersection(test_rolling.columns)
train_rolling = train_rolling[common_columns_rolling]
test_rolling = test_rolling[common_columns_rolling]

# Initialize tracking structures for rolling forecast
feature_usage_count_rolling = {}
model_config_count_rolling = {}
mse_benchmark_all = []
best_predicts_ridge = []
best_predicts_lasso = []
best_mse_ridge_rolling = []
best_mse_lasso_rolling = []
best_alphas_ridge = []
best_alphas_lasso = []
mse_scores_rolling = []
predictions_rolling = []
true_values_rolling = []
predictions_benchmark = []

#block_size = 12  # Define block size for block bootstrap

# Prepare training data with block bootstrap
#train_rolling_bootstrap = block_bootstrap(train_rolling, block_size)

y_train_rolling = train_rolling['INDPRO'].shift(1).iloc[12:]  # Exclude the last period
X_train_rolling = train_rolling.drop('INDPRO', axis=1).iloc[:-12]  # Shift by one period and exclude the first 12 period
#X_train_remain =  train_rolling_bootstrap.drop('INDPRO', axis=1).iloc[]
    
# Rolling forecast process with block bootstrap
for t in range(len(test_rolling) - 12):

    y_test_rolling = test_rolling.iloc[[t + 12]]['INDPRO']  # Predicting y_t+1
    X_test_rolling = test_rolling.iloc[t:t + 1].drop('INDPRO', axis=1)
    
    
    # Get the index of the last training data point
    last_train_index = y_train_rolling.index[-12]

    # Forecast the next value using the benchmark model
    forecast = benchmark_model.predict(start=last_train_index, end=last_train_index)[0]
    
    predictions_benchmark.append(forecast)
    mse_benchmark = mean_squared_error([y_test_rolling], [forecast])
    
    mse_benchmark_all.append(mse_benchmark)
    # Update the model with the new data point
    benchmark_model = AutoReg(y_train_rolling, lags=best_lag).fit()
    
    # Track best MSE and corresponding alpha for Ridge and Lasso
    best_ridge_mse = float('inf')
    best_lasso_mse = float('inf')
    best_ridge_alpha = None
    best_lasso_alpha = None

    # Ridge Regression: Loop through all ridge_alphas
    for alpha in ridge_alphas:
        ridge_model_rolling = Ridge(alpha=alpha)
        ridge_model_rolling.fit(X_train_rolling, y_train_rolling)
        
        # Prediction and MSE
        ridge_prediction_rolling = ridge_model_rolling.predict(X_test_rolling)[0]
        ridge_mse_current_rolling = mean_squared_error([y_test_rolling], [ridge_prediction_rolling])
        
        if ridge_mse_current_rolling < best_ridge_mse:
            best_ridge_mse = ridge_mse_current_rolling
            best_ridge_alpha = alpha
            best_ridge_predict = ridge_prediction_rolling
    # Lasso Regression: Loop through all lasso_alphas
    for alpha in lasso_alphas:
        lasso_model_rolling = Lasso(alpha=alpha)
        lasso_model_rolling.fit(X_train_rolling, y_train_rolling)

        # Prediction and MSE
        lasso_prediction_rolling = lasso_model_rolling.predict(X_test_rolling)[0]
        lasso_mse_current_rolling = mean_squared_error([y_test_rolling], [lasso_prediction_rolling])
        
        if lasso_mse_current_rolling < best_lasso_mse:
            best_lasso_mse = lasso_mse_current_rolling
            best_lasso_alpha = alpha
            best_lasso_predict = lasso_prediction_rolling
            
    # Store best results for Ridge and Lasso
    best_mse_ridge_rolling.append(best_ridge_mse)
    best_alphas_ridge.append(best_ridge_alpha)
    best_predicts_ridge.append(best_ridge_predict)
    
    best_mse_lasso_rolling.append(best_lasso_mse)
    best_alphas_lasso.append(best_lasso_alpha)
    best_predicts_lasso.append(best_lasso_predict)

    # # Optionally, average the predictions from the best Ridge and Lasso models
    # final_prediction_rolling = (ridge_prediction_rolling + lasso_prediction_rolling) / 2
    # predictions_rolling.append(final_prediction_rolling)
    # true_values_rolling.append(y_test_rolling)
    
    # Calculate the absolute correlation with 'INDPRO'
    # Ensure the indices of X_train_rolling and y_train_rolling match
    # X_train_rolling, y_train_rolling = X_train_rolling.align(y_train_rolling, axis=0, join='inner')

    # Calculate the absolute correlation with 'INDPRO'
    correlations_rolling = X_train_rolling.corrwith(y_train_rolling).abs()

    # Sort and take the top 20 most correlated features
    top_features_rolling = correlations_rolling.sort_values(ascending=False).head(40).index

    # Update feature usage count
    for feature in top_features_rolling:
        feature_usage_count_rolling[feature] = feature_usage_count_rolling.get(feature, 0) + 1

    # Update model configuration usage
    model_config_rolling = tuple(sorted(top_features_rolling))
    model_config_count_rolling[model_config_rolling] = model_config_count_rolling.get(model_config_rolling, 0) + 1

    # Restrict X_train and X_test to these top features
    X_train_rolling_bagging = X_train_rolling[top_features_rolling]
    X_test_rolling_bagging = X_test_rolling[top_features_rolling]

    # Initialize and fit the Bagging Regressor
    bagging_model_rolling = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=500, random_state=42)
    bagging_model_rolling.fit(X_train_rolling_bagging, y_train_rolling)

    # Forecast for the current period
    prediction_rolling = bagging_model_rolling.predict(X_test_rolling_bagging)[0]

    # Store predictions and true values
    predictions_rolling.append(prediction_rolling)
    true_values_rolling.append(y_test_rolling)

    # Calculate MSE for the current forecast
    mse_current_rolling = mean_squared_error([y_test_rolling], [prediction_rolling])
    mse_scores_rolling.append(mse_current_rolling)
    
    # DataFrame to store MSE for different feature counts in this iteration
    mse_for_t = {}

    for feature_count in range(1, 21):  # Loop through feature counts from 1 to 20

        # Calculate the absolute correlation with 'INDPRO'
        correlations_rolling = X_train_rolling.corrwith(y_train_rolling).abs()

        # Sort and take the top N most correlated features (N = feature_count)
        top_features_rolling = correlations_rolling.sort_values(ascending=False).head(feature_count).index

        # Update feature usage count
        for feature in top_features_rolling:
            feature_usage_count_rolling[feature] = feature_usage_count_rolling.get(feature, 0) + 1

        # Update model configuration usage
        model_config_rolling = tuple(sorted(top_features_rolling))
        model_config_count_rolling[model_config_rolling] = model_config_count_rolling.get(model_config_rolling, 0) + 1

        # Restrict X_train and X_test to these top features
        X_train_rolling_bagging = X_train_rolling[top_features_rolling]
        X_test_rolling_bagging = X_test_rolling[top_features_rolling]

        # Initialize and fit the Bagging Regressor
        bagging_model_rolling = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=500, random_state=42)
        bagging_model_rolling.fit(X_train_rolling_bagging, y_train_rolling)

        # Forecast for the current period
        prediction_rolling = bagging_model_rolling.predict(X_test_rolling_bagging)[0]

        # Calculate MSE for the current forecast
        mse_current_rolling = mean_squared_error([y_test_rolling.iloc[0]], [prediction_rolling])

        # Store MSE for the current feature count
        mse_for_t[feature_count] = mse_current_rolling

    # Append the MSEs for this iteration (t) to the DataFrame
    mse_feature_count = pd.concat([mse_feature_count, pd.DataFrame([mse_for_t])])
    
    
    # Append actual test observation to training data for next iteration
    # Append actual test observation to training data for the next iteration
    new_row_rolling = test_rolling.iloc[t:t+1]
    train_rolling = pd.concat([train_rolling, new_row_rolling])

    # Append the new test observation for X_train and y_train for the next iteration
    X_train_rolling = pd.concat([X_train_rolling, test_rolling.iloc[t:t + 1].drop('INDPRO', axis=1)])
    y_train_rolling = pd.concat([y_train_rolling, test_rolling.iloc[[t]]['INDPRO']])  # Fixed: No need for index


# Print results
print("Feature Usage Count:")
for feature, count in feature_usage_count_rolling.items():
    print(f"{feature}: {count}")

print("Model Configurations, Their Counts, and Number of Variables Used:")
for config, count in model_config_count_rolling.items():
    num_variables = len(config)  # Count the number of variables in the tuple
    print(f"Configuration: {config}, Count: {count}, Variables used: {num_variables}")

print("\nMSE Scores for Each Prediction Period:")
for i, mse in enumerate(mse_scores_rolling):
    print(f"Period {i+1}: {mse}")

mean_MSE = sum(mse_benchmark_all) / len(mse_benchmark_all)
print(f"The mean of all predicted MSE for the benchmark model is: {mean_MSE}")
mean_MSE = sum(mse_scores_rolling) / len(mse_scores_rolling)
print(f"The mean of all predicted MSE for bagging is: {mean_MSE}")
mean_MSE_ridge = sum(best_mse_ridge_rolling) / len(best_mse_ridge_rolling)
print(f"The mean of all predicted MSE for ridge is: {mean_MSE_ridge}")
mean_MSE_lasso = sum(best_mse_lasso_rolling) / len(best_mse_lasso_rolling)
print(f"The mean of all predicted MSE for lasso is: {mean_MSE_lasso}")

mean_mse_per_feature_count = mse_feature_count.mean()

# Create a new DataFrame to store the results
mean_mse_table = pd.DataFrame({
    'Amount of Features': mean_mse_per_feature_count.index,
    'Mean MSE': mean_mse_per_feature_count.values
})

print(mean_mse_table)

test_rolling = test_rolling[12:]
# Plotting the true values and predictions
# Plot historical data from 2005 to 2014
plt.figure(figsize=(12, 6))
plt.plot(data['2021-01-01':'2023-11-01'].index, data['2021-01-01':'2023-11-01']['INDPRO'], label='True Values (2021-2023)', color='blue')

# Plot true values and predictions from 2015 onwards
plt.plot(test_rolling.index, predictions_benchmark, label='Predictions Benchmark(from 06.2021)', color='purple', linestyle='--')
plt.plot(test_rolling.index, predictions_rolling, label='Predictions Bagging(from 06.2021)', color='red', linestyle='--')
plt.plot(test_rolling.index, best_predicts_ridge, label='Predictions RIDGE(from 06.2021)', color='green', linestyle='--')
plt.plot(test_rolling.index, best_predicts_lasso, label='Predictions LASSO(from 06.2021)', color='orange', linestyle='--')

plt.title('INDPRO: Historical Data and Forecast Comparison for a forecast horizon(y_t+12)')
plt.xlabel('Year')
plt.ylabel('INDPRO Value')
plt.legend()
plt.grid(True)
plt.show()