import pandas as pd
import requests
import certifi
from io import StringIO

file_url = 'https://drive.google.com/uc?export=download&id=1e6E-F_NpGiq5jsApDGhXz2JPb8mIWiWU'

response = requests.get(file_url, verify=certifi.where())
response.raise_for_status()
csv_data = StringIO(response.text)
melbourne_full_df = pd.read_csv(csv_data)

print("Melbourne Housing Full Dataset:")
print(melbourne_full_df.head())

non_numeric_columns = melbourne_full_df.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

melbourne_full_df.info()

columnas = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname']

for col in columnas:
    melbourne_full_df[col] = melbourne_full_df[col].astype('string')

melbourne_full_df['Date']=pd.to_datetime(melbourne_full_df['Date'], format="%d/%m/%Y")
melbourne_full_df.info()

melbourne_full_df.describe()

print("\nMissing values in Melbourne Housing Full Dataset:")
print(melbourne_full_df.isnull().sum())

target_variable = 'Price'
# List all columns
all_columns = melbourne_full_df.columns.tolist()
# Remove target variable to get feature attributes
feature_attributes = [col for col in all_columns if col != target_variable]
print("Target Variable:", target_variable)
print("Feature Attributes:", feature_attributes)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(df):
    numeric_features = df.select_dtypes(include=[np.number])
    for column in numeric_features.columns:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column].dropna())
        plt.title(f'Boxplot of {column}')
        plt.show()

print("Data Distribution in Full Dataset:")
plot_distribution(melbourne_full_df)

melbourne_full_df.isna().sum()

melbourne_full_df.drop_duplicates(inplace=True)
mel_sin_string=melbourne_full_df.copy()
mel_sin_string=mel_sin_string.drop(['Suburb','Address','SellerG'],axis=1)

mel_sin_string[['Method']].value_counts()

mel_sin_string[['Type']].value_counts()

mel_sin_string['CouncilArea'].value_counts()

from sklearn.preprocessing import OrdinalEncoder
oe= OrdinalEncoder()
housing_cat_encoded = oe.fit_transform(mel_sin_string[['Method']])
mel_sin_string['Method_Int'] = housing_cat_encoded
mel_sin_string[['Method_Int']].value_counts()

oe= OrdinalEncoder()
mel_sin_string['CouncilArea'].replace(['NAType', 'str'], None)
mel_sin_string['CouncilArea'] = mel_sin_string['CouncilArea'].astype(str)
housing_cat_encoded = oe.fit_transform(mel_sin_string[['CouncilArea']])
mel_sin_string['CouncilArea_Int'] = housing_cat_encoded
mel_sin_string[['CouncilArea_Int']].value_counts()

oe= OrdinalEncoder()
mel_sin_string['Regionname'].replace(['NAType', 'str'], None)
mel_sin_string['Regionname'] = mel_sin_string['Regionname'].astype(str)
housing_cat_encoded = oe.fit_transform(mel_sin_string[['Regionname']])
mel_sin_string['Regionname_Int'] = housing_cat_encoded
mel_sin_string[['Regionname_Int']].value_counts()

mel_sin_string=mel_sin_string[mel_sin_string['Type']=='h']
dfmelboune=mel_sin_string.drop(['Type','Method','CouncilArea','Regionname'],axis=1)
dfmelboune

corr_matrix= dfmelboune.corr()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(16, 12))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.1)
    plt.title('Correlation Matrix')
    plt.show()
plot_correlation_heatmap(corr_matrix)

def plot_pairplot(df, features):
    sns.pairplot(df[features])
    plt.show()

selected_features = ['Rooms', 'Bathroom', 'BuildingArea', 'Landsize', 'YearBuilt']

print("Pairplot for Selected Features in Full Dataset:")
plot_pairplot(dfmelboune, selected_features)

print("Correlation with Price (Full Dataset):")
print(corr_matrix['Price'].sort_values(ascending=False))

selected_features_full = corr_matrix['Price'].sort_values(ascending=False).index[1:5].tolist()

print("Selected Features for Detailed Analysis:", selected_features_full)

def plot_feature_relationships(df, features, target):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature, y=target)
        plt.title(f'Relationship between {feature} and {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.show()

plot_feature_relationships(melbourne_full_df, selected_features_full, 'Price')

import numpy as np
from sklearn.model_selection import train_test_split

selected_features_full = ['Rooms', 'Bathroom', 'BuildingArea', 'Landsize', 'YearBuilt']
melbourne_full_df = melbourne_full_df.dropna(subset=selected_features_full + ['Price'])

melbourne_full_df['Landsize'] = np.log1p(melbourne_full_df['Landsize'])
melbourne_full_df['Price'] = np.log1p(melbourne_full_df['Price'])

X = melbourne_full_df[selected_features_full]
y = melbourne_full_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Training feature matrix shape: {X_train.shape}')
print(f'Test feature matrix shape: {X_test.shape}')
print(f'Training target vector shape: {y_train.shape}')
print(f'Test target vector shape: {y_test.shape}')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

y_train_pred = ols_model.predict(X_train)
y_test_pred = ols_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("OLS Linear Regression Model")
print("---------------------------")
print(f"Coefficients: {ols_model.coef_}")
print(f"Intercept: {ols_model.intercept_}")
print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R-squared: {train_r2}")
print(f"Test R-squared: {test_r2}")

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

param_grid = {
    'sgd__alpha': [0.0001, 0.001, 0.01, 0.1],
    'sgd__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'sgd__eta0': [0.0001, 0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_sgd_model = grid_search.best_estimator_

y_train_pred_sgd = best_sgd_model.predict(X_train)
y_test_pred_sgd = best_sgd_model.predict(X_test)

train_mse_sgd = mean_squared_error(y_train, y_train_pred_sgd)
test_mse_sgd = mean_squared_error(y_test, y_test_pred_sgd)
train_r2_sgd = r2_score(y_train, y_train_pred_sgd)
test_r2_sgd = r2_score(y_test, y_test_pred_sgd)

print("Tuned SGD Linear Regression Model")
print("---------------------------")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Coefficients: {best_sgd_model.named_steps['sgd'].coef_}")
print(f"Intercept: {best_sgd_model.named_steps['sgd'].intercept_}")
print(f"Training MSE: {train_mse_sgd}")
print(f"Test MSE: {test_mse_sgd}")
print(f"Training R-squared: {train_r2_sgd}")
print(f"Test R-squared: {test_r2_sgd}")


#%%
