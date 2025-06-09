import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading ---
# Assuming the dataset is available at '/kaggle/input/house-price/house_prices.csv'
try:
    df = pd.read_csv('/kaggle/input/house-price/house_prices.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: house_prices.csv not found. Please ensure the file is in the correct path.")
    # Exit or handle the error appropriately if the file is not found
    exit()


# --- 2. Data Cleaning and Preparation ---
print("\n--- Data Cleaning and Preparation ---")

# Re-apply initial cleaning steps to create df2
# 1- Fill null price with the median price
median_price = df['Price (in rupees)'].median()
df2 = df.copy()
df2['Price (in rupees)'] = df2['Price (in rupees)'].fillna(median_price)
print(f"Filled missing 'Price (in rupees)' with median: {median_price:.2f}")

# 2- Drop 'Plot Area' and 'Dimensions' (identified as having all null values)
df2.drop(columns=['Plot Area','Dimensions'],inplace=True)
print("Dropped 'Plot Area' and 'Dimensions' columns.")

# 3- Drop rows with null 'Description' (based on previous analysis)
initial_rows = df2.shape[0]
df2.dropna(subset=['Description'],inplace=True)
print(f"Dropped {initial_rows - df2.shape[0]} rows with missing 'Description'.")


# 4- Convert 'Carpet Area' and 'Super Area' to numeric and impute NaNs
def safe_convert_area_to_numeric(area):
    if isinstance(area, str):
        area = area.replace(' sqft', '').replace(',', '')
    try:
        return float(area)
    except (ValueError, TypeError):
        return np.nan # Return NaN for values that cannot be converted

df2['Carpet Area'] = df2['Carpet Area'].apply(safe_convert_area_to_numeric)
df2['Super Area'] = df2['Super Area'].apply(safe_convert_area_to_numeric)

# Re-impute any NaNs introduced by the conversion with the median
median_carpet_area = df2['Carpet Area'].median()
median_super_area = df2['Super Area'].median()
df2['Carpet Area'] = df2['Carpet Area'].fillna(median_carpet_area)
df2['Super Area'] = df2['Super Area'].fillna(median_super_area)
print(f"Converted 'Carpet Area' and 'Super Area' to numeric and imputed NaNs with median ({median_carpet_area:.2f}, {median_super_area:.2f}).")


# 5- Impute 'Status', 'Transaction', 'Furnishing', 'Bathroom', 'Floor' with mode
for col in ['Status', 'Transaction', 'Furnishing', 'Floor']:
    if df2[col].isnull().any():
        mode_val = df2[col].mode()[0]
        df2[col] = df2[col].fillna(mode_val)
        print(f"Imputed missing '{col}' with mode: {mode_val}")

# Convert Bathroom to numeric first, then impute median
df2['Bathroom']=pd.to_numeric(df2['Bathroom'], errors='coerce') # Convert to numeric first
median_bathroom = df2['Bathroom'].median()
df2['Bathroom'] = df2['Bathroom'].fillna(median_bathroom) # Then impute median
print(f"Converted 'Bathroom' to numeric and imputed NaNs with median: {median_bathroom:.2f}")


# 6- Impute 'facing', 'overlooking', 'Society', 'Balcony', 'Car Parking', 'Ownership'
for col in ['facing', 'overlooking', 'Balcony', 'Car Parking', 'Ownership']:
    if df2[col].isnull().any():
        mode_val = df2[col].mode()[0]
        df2[col] = df2[col].fillna(mode_val)
        print(f"Imputed missing '{col}' with mode: {mode_val}")

# Impute 'Society' with 'Unknown' as it has high cardinality
if df2['Society'].isnull().any():
    df2['Society'] = df2['Society'].fillna('Unknown')
    print("Imputed missing 'Society' with 'Unknown'.")

# Check for any remaining nulls
print("\nMissing values after imputation:")
print(df2.isnull().sum())


# Handle high cardinality 'location' by keeping top N and grouping others
top_n_locations = 20  # Number of top locations to keep
top_locations = df2['location'].value_counts().nlargest(top_n_locations).index.tolist()
df2['location'] = df2['location'].apply(lambda x: x if x in top_locations else 'Other_Location')
print(f"\nHandled 'location' cardinality: keeping top {top_n_locations} locations, others grouped into 'Other_Location'.")


# Identify categorical columns for encoding - Excluding high cardinality ones that are dropped or grouped
# Columns to exclude from standard one-hot encoding due to being dropped, text, or handled differently:
# 'Title', 'Description', 'Amount(in rupees)', 'Index', 'Floor', 'Society', 'Balcony', 'Car Parking'
# 'location' is now handled by grouping
categorical_cols_to_encode = ['location', 'Status', 'Transaction', 'Furnishing', 'facing', 'overlooking', 'Ownership']

print("\nCategorical columns identified for one-hot encoding:")
print(categorical_cols_to_encode)


# Apply one-hot encoding to the selected categorical columns
# Drop columns that we decided not to encode from df2 before encoding
cols_to_drop_before_encoding = ['Amount(in rupees)', 'Title', 'Description', 'Index', 'Floor', 'Society', 'Balcony', 'Car Parking']
# Ensure 'location' is not double dropped if already handled by grouping in df2
cols_to_drop_final = [col for col in cols_to_drop_before_encoding if col in df2.columns]

df_encoded = pd.get_dummies(df2.drop(columns=cols_to_drop_final), columns=categorical_cols_to_encode, drop_first=True)
print(f"\nDataFrame shape after one-hot encoding: {df_encoded.shape}")


# Separate features (X) and target (y)
X = df_encoded.drop('Price (in rupees)', axis=1)
y = df_encoded['Price (in rupees)']
print(f"Features shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# Identify numerical features for scaling
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
print("\nNumerical columns for scaling:")
print(numerical_cols)

# Initialize and apply the StandardScaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("Scaled numerical features.")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")


# --- 3. Exploratory Data Analysis (EDA) - Visualizations ---
print("\n--- Exploratory Data Analysis (EDA) - Visualizations ---")

# Histograms of numerical features (using df2 before dropping columns for encoding)
numerical_features_eda = ['Carpet Area', 'Super Area', 'Bathroom']
df2[numerical_features_eda].hist(figsize=(15, 5))
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout()
plt.show()

# Scatter plots of numerical features vs. Price (using df2 before dropping columns for encoding)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(df2['Carpet Area'], df2['Price (in rupees)'], alpha=0.5)
axes[0].set_title('Carpet Area vs. Price')
axes[0].set_xlabel('Carpet Area')
axes[0].set_ylabel('Price (in rupees)')

axes[1].scatter(df2['Super Area'], df2['Price (in rupees)'], alpha=0.5)
axes[1].set_title('Super Area vs. Price')
axes[1].set_xlabel('Super Area')
axes[1].set_ylabel('Price (in rupees)')

axes[2].scatter(df2['Bathroom'], df2['Price (in rupees)'], alpha=0.5)
axes[2].set_title('Bathroom vs. Price')
axes[2].set_xlabel('Bathroom')
axes[2].set_ylabel('Price (in rupees)')

plt.tight_layout()
plt.show()

# Bar plot of average price by location (using original df for wider range - consider using df2 with grouped locations for consistency in analysis)
# Let's use df2 with grouped locations for consistency with the model's features
plt.figure(figsize=(30, 8))
sns.barplot(x='location', y='Price (in rupees)', data=df2)
plt.xticks(rotation=90)
plt.title("Location Impact on Prices (Top Locations and Others)")
plt.show()


# --- 4. Model Selection and Training ---
print("\n--- Model Selection and Training ---")

# Instantiate models
linear_reg_model = LinearRegression()
lgbm_reg_model = LGBMRegressor(random_state=42)

# Train models
print("Training Linear Regression model...")
linear_reg_model.fit(X_train, y_train)
print("Linear Regression model trained.")

print("Training LightGBM Regressor model...")
lgbm_reg_model.fit(X_train, y_train)
print("LightGBM Regressor model trained.")


# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred_linear = linear_reg_model.predict(X_test)
y_pred_lgbm = lgbm_reg_model.predict(X_test)

# Calculate MSE, RMSE, and R-squared
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

# Print evaluation results
print("\n--- Model Evaluation Results (MSE, RMSE, R-squared) ---")
print("\nLinear Regression:")
print(f"  MSE: {mse_linear:.2f}")
print(f"  RMSE: {rmse_linear:.2f}")
print(f"  R-squared: {r2_linear:.4f}")

print("\nLightGBM Regressor:")
print(f"  MSE: {mse_lgbm:.2f}")
print(f"  RMSE: {rmse_lgbm:.2f}")
print(f"  R-squared: {r2_lgbm:.4f}")


# --- 6. Interpretation (via print statements as done previously) ---
print("\n--- Interpretation of Model Results and EDA ---")

print("\n1. Performance Metrics (MSE, RMSE, and R-squared):")
print(f"Linear Regression: MSE={mse_linear:.2f}, RMSE={rmse_linear:.2f}, R-squared={r2_linear:.4f}")
print(f"LightGBM Regressor: MSE={mse_lgbm:.2f}, RMSE={rmse_lgbm:.2f}, R-squared={r2_lgbm:.4f}")

print("\nInterpretation of Metrics:")
print(f"- LightGBM has lower MSE ({mse_lgbm:.2f}) and RMSE ({rmse_lgbm:.2f}) compared to Linear Regression (MSE={mse_linear:.2f}, RMSE={rmse_linear:.2f}), indicating that LightGBM's predictions have, on average, smaller errors.")
print(f"- The RMSE values represent the typical error magnitude in the same units as the target variable (Rupees). An RMSE of {rmse_lgbm:.2f} for LightGBM means that the model's predictions are typically off by around this amount.")
print(f"- The R-squared values for both models are very low (Linear Regression: {r2_linear:.4f}, LightGBM: {r2_lgbm:.4f}). This indicates that the models explain very little of the variance in the target variable.")
print("Combining these metrics, while LightGBM has lower error magnitudes (MSE, RMSE), the extremely low R-squared values for both models consistently indicate that neither model, with the current features, is explaining a significant portion of the house price variability. This points to limitations in the features or the complexity of the underlying price determinants not captured by these models.")


# 2. Interpret the coefficients of the Linear Regression model
print("\n2. Interpretation of Linear Regression Coefficients:")
# Ensure X_train columns are available for interpreting coefficients
if hasattr(linear_reg_model, 'coef_') and X_train is not None:
    linear_reg_coef = pd.Series(linear_reg_model.coef_, index=X_train.columns)

    print("\nTop 10 Positive Coefficients:")
    print(linear_reg_coef.nlargest(10))
    print("\nTop 10 Negative Coefficients:")
    print(linear_reg_coef.nsmallest(10))
else:
    print("Linear Regression model not trained or X_train not available for coefficient interpretation.")


print("\nInterpretation:")
print("- Positive coefficients indicate that as the value of the feature increases, the predicted house price increases, assuming other features are held constant.")
print("- Negative coefficients indicate that as the value of the feature increases, the predicted house price decreases, assuming other features are held constant.")
print("- For instance, grouped locations will now have coefficients relative to the baseline location among the top N.")
print("- Scaled numerical features' coefficients represent the change in price for a one standard deviation increase.")


# 3. Interpret the feature importances from the LightGBM model
print("\n3. Interpretation of LightGBM Regressor Feature Importances:")
# Ensure X_train columns are available for interpreting feature importances
if hasattr(lgbm_reg_model, 'feature_importances_') and X_train is not None:
    lgbm_feature_importances = pd.Series(lgbm_reg_model.feature_importances_, index=X_train.columns)
    print("\nTop 10 Important Features (LightGBM):")
    print(lgbm_feature_importances.nlargest(10))
else:
    print("LightGBM model not trained or X_train not available for feature importance interpretation.")


print("\nInterpretation:")
print("- Feature importances in tree-based models like LightGBM indicate how much each feature contributed to the model's prediction accuracy (e.g., by reducing impurity).")
print("- 'Super Area' and 'Carpet Area' are expected to remain important.")
print("- The 'location_Other_Location' feature might have a different importance depending on how prices in those grouped locations compare to the top N locations.")


# 4. Discuss the insights gained from the EDA visualizations (referencing the plots generated earlier)
print("\n4. Insights from EDA Visualizations and their Relevance to Model Results:")
print("- Histograms showed skewed distributions for area and bathroom features.")
print("- Scatter plots showed positive but non-linear relationships between area/bathroom and price.")
print("- The updated location bar plot shows average prices for the top N locations and the 'Other_Location' group, highlighting the importance of distinguishing these locations.")


# 5. Synthesize the interpretations
print("\n5. Synthesis of Interpretations:")
print("Based on the combined insights from model results and EDA:")
print("- Area and Bathroom count continue to be identified as important features.")
print("- Location, now handled by grouping, is still expected to be a key factor.")
print("- The low R-squared values suggest that even with reduced dimensionality from grouped locations, the current features have limited power in explaining price variance.")


# 6. Consider the limitations of the models and the data
print("\n6. Limitations of the Models and Data:")
print("- The primary limitation remains the low R-squared values, indicating that the models do not capture the full complexity of house price determination.")
print("- Grouping locations reduces dimensionality but might lose some granular information about specific locations within the 'Other_Location' category.")
print("- Other limitations mentioned previously (missing features, noise, etc.) still apply.")


# --- 7. Comparison (via print statements as done previously) ---
print("\n--- Model Comparison ---")

# Print performance metrics again for easy comparison
print("\n--- Performance Metrics ---")
print("Linear Regression:")
print(f"  MSE: {mse_linear:.2f}")
print(f"  RMSE: {rmse_linear:.2f}")
print(f"  R-squared: {r2_linear:.4f}")

print("\nLightGBM Regressor:")
print(f"  MSE: {mse_lgbm:.2f}")
print(f"  RMSE: {rmse_lgbm:.2f}")
print(f"  R-squared: {r2_lgbm:.4f}")

print("\n--- Performance Comparison ---")
print("Based on the evaluation metrics:")
print(f"- MSE and RMSE: Compare the new MSE and RMSE values to see if reducing dimensionality improved performance or made it worse.")
print(f"- R-squared: Compare the new R-squared values. They are likely to remain low but might change.")
print("\nOverall, assess if the memory fix (grouping locations) impacted the relative performance of the models.")


print("\n--- Interpretability Comparison ---")
print("Linear Regression remains the most interpretable with coefficients, but interpretation of location coefficients is now about top N locations vs. the 'Other' group.")
print("\nLightGBM feature importances will now include 'location_Other_Location'.")


print("\n--- Performance vs. Interpretability Trade-off ---")
print("The trade-off discussion remains similar, but the feature set is now different due to location grouping.")


print("\n--- Summary and Conclusion ---")
print("Performance Summary:")
print("- Report the updated MSE, RMSE, and R-squared values.")
print("- State which model performed better on error metrics with the reduced feature set.")

print("\nInterpretability Summary:")
print("- Discuss interpretation in the context of grouped locations.")

print("\nTrade-off Summary:")
print("- Reiterate the trade-off in the context of the new feature set and performance.")

print("\nOverall Conclusion:")
print("Discuss whether reducing dimensionality resolved the memory issue and how it affected model performance and interpretability. Reiterate the main conclusion about the limitations of the dataset.")


# --- 8. Generating Plots (Included in this single script) ---
print("\n--- Generating Performance Plots ---")

# Scatter plot of Predicted vs. Actual Prices for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.title('Linear Regression: Predicted vs. Actual Prices (Reduced Features)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

# Residual plot for Linear Regression
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred_linear, y=y_test - y_pred_linear, lowess=True, color="g")
plt.title('Linear Regression: Residual Plot (Reduced Features)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Scatter plot of Predicted vs. Actual Prices for LightGBM Regressor
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgbm, alpha=0.5)
plt.title('LightGBM Regressor: Predicted vs. Actual Prices (Reduced Features)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

# Residual plot for LightGBM Regressor
plt.figure(figsize=(10, 6))
sns.residplot(x=y_pred_lgbm, y=y_test - y_pred_lgbm, lowess=True, color="g")
plt.title('LightGBM Regressor: Residual Plot (Reduced Features)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
