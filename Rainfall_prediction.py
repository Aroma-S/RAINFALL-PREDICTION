#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[27]:


# Load the data from CSV
data = pd.read_csv("C:/Users/Aroma/Desktop/TN_rainfall.csv")
data


# In[28]:


# Data Preprocessing and Cleaning
# Remove any rows with missing values
data = data.dropna()
data


# In[29]:


subdivs = data['District'].unique()
num_of_subdivs = subdivs.size
print('Total # of Subdivs: ' + str(num_of_subdivs))
subdivs


# In[69]:



from sklearn.preprocessing import MinMaxScaler

# Select features for normalization (excluding columns like 'Year' and 'Districts' if they exist)
features_to_normalize = ['Actual South West Monsoon', 
                         'Actual North East Monsoon',
                         'Actual Winter Season','Acutal Hot Weather Season']

# Extract selected features for normalization
features = data[features_to_normalize]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the features
normalized_features = scaler.fit_transform(features)

# Replace the original features in the dataframe with normalized features
data[features_to_normalize] = normalized_features

# Now, data contains normalized features
print(data.head())  # Display the first few rows of the normalized data


# In[71]:


import scipy.stats as stats

# Choose a specific column to check for normality (e.g., 'Total Actual Rainfall in mm')
column_to_check = 'Actual South West Monsoon'
    

# Plotting Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data[column_to_check], dist="norm", plot=plt)
plt.title(f'Q-Q Plot for {column_to_check}')
plt.show()

# Perform Shapiro-Wilk test for normality
shapiro_test_statistic, shapiro_p_value = stats.shapiro(data[column_to_check])
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'P-value: {shapiro_p_value}')

# Interpret the results
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'The {column_to_check} data appears to be normally distributed (failed to reject H0)')
else:
    print(f'The {column_to_check} data does not appear to be normally distributed (reject H0)')


# In[73]:


column_to_check = 'Actual North East Monsoon'
    

# Plotting Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data[column_to_check], dist="norm", plot=plt)
plt.title(f'Q-Q Plot for {column_to_check}')
plt.show()

# Perform Shapiro-Wilk test for normality
shapiro_test_statistic, shapiro_p_value = stats.shapiro(data[column_to_check])
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'P-value: {shapiro_p_value}')

# Interpret the results
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'The {column_to_check} data appears to be normally distributed (failed to reject H0)')
else:
    print(f'The {column_to_check} data does not appear to be normally distributed (reject H0)')


# In[77]:



# Choose a specific column to transform (e.g., 'Total Actual Rainfall in mm')
column_to_transform = 'Actual North East Monsoon'

# Perform square root transformation
transformed_data = np.sqrt(data[column_to_transform])

# Plotting Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(transformed_data, dist="norm", plot=plt)
plt.title(f'Q-Q Plot for Square Root-transformed {column_to_transform}')
plt.show()

# Perform Shapiro-Wilk test for normality on transformed data
shapiro_test_statistic, shapiro_p_value = stats.shapiro(transformed_data)
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'P-value: {shapiro_p_value}')

# Interpret the results
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'The square root-transformed {column_to_transform} data appears to be normally distributed (failed to reject H0)')
else:
    print(f'The square root-transformed {column_to_transform} data does not appear to be normally distributed (reject H0)')


# In[78]:


column_to_check = 'Actual Winter Season'
    

# Plotting Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data[column_to_check], dist="norm", plot=plt)
plt.title(f'Q-Q Plot for {column_to_check}')
plt.show()

# Perform Shapiro-Wilk test for normality
shapiro_test_statistic, shapiro_p_value = stats.shapiro(data[column_to_check])
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'P-value: {shapiro_p_value}')

# Interpret the results
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'The {column_to_check} data appears to be normally distributed (failed to reject H0)')
else:
    print(f'The {column_to_check} data does not appear to be normally distributed (reject H0)')


# In[84]:



# Choose a specific column to transform (e.g., 'Actual Rainfall in Winter Season in mm')
column_to_transform = 'Actual Winter Season'

# Remove outliers (considering values outside 1.5*IQR as outliers)
Q1 = data[column_to_transform].quantile(0.25)
Q3 = data[column_to_transform].quantile(0.75)
IQR = Q3 - Q1
data_without_outliers = data[(data[column_to_transform] >= Q1 - 1.5*IQR) & (data[column_to_transform] <= Q3 + 1.5*IQR)]

# Apply inverse hyperbolic sine transformation
transformed_data = np.arcsinh(data_without_outliers[column_to_transform])

# Plotting Q-Q plot for transformed data
plt.figure(figsize=(8, 6))
stats.probplot(transformed_data, dist="norm", plot=plt)
plt.title(f'Q-Q Plot for Transformed {column_to_transform}')
plt.show()

# Perform Shapiro-Wilk test for normality on transformed data
shapiro_test_statistic, shapiro_p_value = stats.shapiro(transformed_data)
print(f'Shapiro-Wilk Test Statistic: {shapiro_test_statistic}')
print(f'P-value: {shapiro_p_value}')

# Interpret the results
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'The transformed {column_to_transform} data appears to be normally distributed (failed to reject H0)')
else:
    print(f'The transformed {column_to_transform} data does not appear to be normally distributed (reject H0)')


# In[32]:


print("Basic Statistics:")
print(data.describe())


# In[56]:


from matplotlib import rcParams

# Set the figure size globally
rcParams['figure.figsize'] = 16, 12

# Group data by 'Year' and 'Districts', then sum the 'Total Actual Rainfall in mm'
district_wise_annual_rainfall = data.groupby(['Year', 'District'])['Actual Annual Total'].sum().unstack()

# Plotting the line graph with a larger image
plt.figure(figsize=(14, 10))  # Adjust the width and height as needed
district_wise_annual_rainfall.T.plot(marker='o', linewidth=2)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Annual Rainfall (mm)', fontsize=14)
plt.title('District-wise Annual Rainfall Distribution (2018-2020)', fontsize=16)
plt.legend(title='Districts', bbox_to_anchor=(1, 1), loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()


plt.show()


# In[38]:


plt.figure(figsize=(10, 8))

# Example: Correlation matrix with larger plot
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=1, linecolor='white', square=True)
plt.title('Correlation Matrix', fontsize=16)

plt.show()


# In[43]:


# Calculate total annual rainfall for each year
total_rainfall_per_year = data.groupby('Year')['Actual Annual Total'].sum()

# Print the result
print("Total Annual Rainfall for Each Year:")
print(total_rainfall_per_year)


# In[66]:


from sklearn.metrics import mean_squared_error
# Choose features (X) and target (y)
features = data[["Actual South West Monsoon", 
                 "Actual North East Monsoon",
                 "Actual Winter Season","Acutal Hot Weather Season"]]
target = data['Actual Annual Total']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)


# Plotting actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Rainfall (mm)')
plt.ylabel('Predicted Rainfall (mm)')
plt.title('Actual vs. Predicted Rainfall')
plt.grid(True)
plt.show()

residuals = y_test - predictions
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals)
plt.xlabel('Predicted Rainfall (mm)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='--')  # Add horizontal line at y=0 for reference
plt.grid(True)
plt.show()


# In[65]:


# Calculate Mean Absolute Deviation (MAD)
mad = abs(y_test - predictions).mean()
print('Mean Absolute Deviation (MAD):', mad)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error (MSE):', mse)

# Calculate Root Mean Squared Error (RMSE)
rmse = mse ** 0.5
print('Root Mean Squared Error (RMSE):', rmse)


# In[67]:


from sklearn.metrics import r2_score

# Assuming y_test contains the actual values and predictions contains the predicted values
# Calculate R^2 score
r2 = r2_score(y_test, predictions)

print('R^2 Score:', r2)


# In[ ]:





# In[ ]:




