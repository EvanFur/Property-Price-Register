import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import joblib # To potentially load scaler/encoder if needed for prediction on new data later

# --- Load Pre-processed Data ---
# Load the features and target variables saved by the preprocessing script
try:
    X_train = pd.read_csv('property_X_train.csv')
    y_train_original = pd.read_csv('property_y_train.csv').squeeze("columns") # Use squeeze to get Series
    X_test = pd.read_csv('property_X_test.csv')
    y_test_original = pd.read_csv('property_y_test.csv').squeeze("columns") # Use squeeze to get Series
    print("Training and testing data loaded successfully.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_original.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_original.shape}")

    # Optional: Load the scaler and encoder if you need them later
    # scaler = joblib.load('property_scaler.joblib')
    # encoder = joblib.load('property_encoder.joblib')

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please ensure 'property_preprocessing_revised' script ran successfully and files exist.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Target Transformation (Log Transform) ---
# Apply log transformation to the loaded target variable (y)
# Using log1p (log(1+x)) to handle any potential zero values if they exist
y_train_log = np.log1p(y_train_original)
y_test_log = np.log1p(y_test_original)

# --- Visualize Price Distributions (Optional but Recommended) ---
# Create a figure to compare original vs log-transformed target distributions (using test set as example)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Original price distribution (using y_test_original)
sns.histplot(y_test_original, kde=True, ax=ax1, bins=50)
ax1.set_title('Original Property Price Distribution (Test Set)')
ax1.set_xlabel('Price (€)')
ax1.set_ylabel('Count')

# Log-transformed price distribution (using y_test_log)
sns.histplot(y_test_log, kde=True, ax=ax2, bins=50)
ax2.set_title('Log-Transformed Property Price Distribution (Test Set)')
ax2.set_xlabel('Log(Price + 1)')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('price_transformation_comparison_loaded.png')
print("\nSaved plot: price_transformation_comparison_loaded.png")
# plt.show() # Uncomment to display plot immediately
plt.close() # Close plot


# --- Model Training ---
# Initialize the Random Forest Regressor model
# You can tune hyperparameters like n_estimators, max_depth, etc.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use n_jobs=-1 to use all cores

print("\nTraining Random Forest Regressor model...")
# Train the model using the pre-processed training features (X_train)
# and the log-transformed training target (y_train_log)
model.fit(X_train, y_train_log)
print("Model training complete.")

# --- Make Predictions ---
# Make predictions on the pre-processed test features (X_test)
# Predictions will be on the log scale
y_pred_log = model.predict(X_test)

# --- Inverse Transform Predictions ---
# Convert log-scale predictions back to the original price scale
# Use expm1 which is the inverse of log1p
y_pred_original = np.expm1(y_pred_log)

# Note: y_test_original is already available from loading the data

# --- Evaluate the Model ---
print("\nEvaluating model performance...")

# Calculate metrics on the log scale
try:
    rmse_log = math.sqrt(mean_squared_error(y_test_log, y_pred_log))
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    r2_log = r2_score(y_test_log, y_pred_log)

    print(f"\nModel performance on log-transformed scale:")
    print(f"RMSE (log scale): {rmse_log:.4f}")
    print(f"MAE (log scale): {mae_log:.4f}")
    print(f"R² (log scale): {r2_log:.4f}")
except Exception as e:
    print(f"Error calculating metrics on log scale: {e}")


# Calculate metrics on the original price scale
try:
    # Ensure no negative predictions after inverse transform (can happen with poor models/log transform)
    y_pred_original[y_pred_original < 0] = 0

    rmse_original = math.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mae_original = mean_absolute_error(y_test_original, y_pred_original)
    r2_original = r2_score(y_test_original, y_pred_original)

    print(f"\nModel performance on original price scale:")
    print(f"RMSE (€): {rmse_original:.2f}")
    print(f"MAE (€): {mae_original:.2f}")
    print(f"R² (original scale): {r2_original:.4f}")
except Exception as e:
    print(f"Error calculating metrics on original scale: {e}")


# --- Visualize Actual vs Predicted Prices ---
plt.figure(figsize=(10, 10)) # Make plot square
# Use a sample if the test set is very large to avoid overplotting
sample_size = min(5000, len(y_test_original))
indices = np.random.choice(len(y_test_original), sample_size, replace=False)

plt.scatter(y_test_original.iloc[indices], y_pred_original[indices], alpha=0.3, s=10) # Smaller points, lower alpha

# Add the y=x line (perfect prediction line)
min_val = min(y_test_original.min(), y_pred_original.min())
max_val = max(y_test_original.max(), y_pred_original.max())
# Adjust limits slightly beyond min/max for better visualization
plot_min = max(0, min_val * 0.95) # Ensure min is not negative
plot_max = max_val * 1.05
plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Price (€)')
plt.ylabel('Predicted Price (€)')
plt.title('Actual vs Predicted Property Prices (Original Scale)')
plt.xlim(plot_min, plot_max)
plt.ylim(plot_min, plot_max)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('price_prediction_results_revised.png')
print("\nSaved plot: price_prediction_results_revised.png")
# plt.show() # Uncomment to display plot immediately
plt.close()

# Optional: Feature Importance
try:
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance_df.head(20))

    # Plot top N feature importances
    plt.figure(figsize=(10, 8))
    n_top_features = 20
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(n_top_features), palette='viridis')
    plt.title(f'Top {n_top_features} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    print("\nSaved plot: feature_importances.png")
    # plt.show()
    plt.close()
except Exception as e:
    print(f"\nCould not calculate/plot feature importances: {e}")


print("\nModel training, prediction, and evaluation complete!")

