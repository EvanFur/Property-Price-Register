import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import re # Import regex library for cleaner extraction

# --- Configuration ---
DATE_COLUMN = 'Date of Sale (dd/mm/yyyy)'
# Attempt to find the price column dynamically, but provide a fallback/check
PRICE_COLUMN_CANDIDATES = [
    'Price (€)', # Common format
    'SALE PRICE',
    'Price (VAT exclusive)'
    # Add other potential names if needed
]
# Find the actual price column name present in the DataFrame
price_col_actual = None
df_peek = pd.read_csv("PPR-ALL.csv", encoding="Windows-1252", nrows=1) # Peek at header
for candidate in PRICE_COLUMN_CANDIDATES:
    if candidate in df_peek.columns:
        price_col_actual = candidate
        break
# Fallback: Search for any column containing 'Price' if specific candidates fail
if price_col_actual is None:
    potential_cols = [col for col in df_peek.columns if 'Price' in col]
    if potential_cols:
        price_col_actual = potential_cols[0] # Take the first match
    else:
        raise ValueError("Could not automatically determine the Price column. Please set PRICE_COLUMN_CANDIDATES correctly.")

print(f"Using Price Column: {price_col_actual}")

COUNTY_COLUMN = 'County'
PROPERTY_DESC_COLUMN = 'Description of Property'
PROPERTY_SIZE_COLUMN = 'Property Size Description'
NOT_FULL_MARKET_PRICE_COLUMN = 'Not Full Market Price'
VAT_EXCLUSIVE_COLUMN = 'VAT Exclusive'
EIRCODE_COLUMN = 'Eircode' # Assuming this column exists based on original code
TARGET_VARIABLE = 'Price' # We will predict the original price (or its log transform later)

# --- Load Data ---
try:
    df = pd.read_csv("PPR-ALL.csv", encoding="Windows-1252", low_memory=False)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: PPR-ALL.csv not found. Make sure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    print("Trying with utf-8 encoding...")
    try:
        df = pd.read_csv("PPR-ALL.csv", encoding="utf-8", low_memory=False)
        print(f"Dataset loaded successfully with utf-8. Shape: {df.shape}")
    except Exception as e_utf8:
        print(f"Error loading CSV with utf-8: {e_utf8}")
        exit()

# --- Basic Exploration (Optional) ---
print("\nFirst few rows:\n", df.head())
# print("\nData types:\n", df.dtypes)
# print("\nColumns:\n", df.columns.tolist())
# print("\nMissing values:\n", df.isnull().sum())

# --- Data Cleaning ---

# 1. Date Parsing
try:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='%d/%m/%Y')
except KeyError:
    print(f"Error: Date column '{DATE_COLUMN}' not found.")
    exit()
except Exception as e:
    print(f"Error parsing date column: {e}")
    # Consider alternative formats or error handling if needed
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')


# 2. Clean Price Column
if price_col_actual not in df.columns:
     print(f"Error: Determined price column '{price_col_actual}' not found in loaded data.")
     exit()

# Convert to string first to ensure .str methods work
df[price_col_actual] = df[price_col_actual].astype(str)
# Remove currency symbols, commas, and potential encoding artifacts
df[price_col_actual] = df[price_col_actual].str.replace(r'[€,â]', '', regex=True)
# Convert to numeric, coercing errors to NaN
df[TARGET_VARIABLE] = pd.to_numeric(df[price_col_actual], errors='coerce')
# Drop the original potentially messy price column if it's different from TARGET_VARIABLE
if price_col_actual != TARGET_VARIABLE:
     df = df.drop(columns=[price_col_actual])

print(f"\nMissing prices after cleaning: {df[TARGET_VARIABLE].isnull().sum()}")
# Optional: Drop rows with missing target variable (essential for training)
df.dropna(subset=[TARGET_VARIABLE], inplace=True)
print(f"Shape after dropping missing prices: {df.shape}")


# 3. Convert Boolean Columns ('Yes'/'No' to True/False)
def map_boolean(col_name):
    if col_name in df.columns:
        original_type = df[col_name].dtype
        # Handle potential non-string types before mapping
        df[col_name] = df[col_name].astype(str).str.lower().map({'yes': True, 'no': False})
        # If mapping results in NaNs (due to unexpected values), decide how to handle them.
        # Option 1: Fill with False (assuming 'No' is the default/safer option)
        df[col_name].fillna(False, inplace=True)
        # Option 2: Fill with a specific category or drop rows
        # df[col_name].fillna('Unknown', inplace=True) # Example
        # df.dropna(subset=[col_name], inplace=True) # Example
        print(f"Processed boolean column: {col_name}")
    else:
        print(f"Warning: Boolean column '{col_name}' not found.")

map_boolean(NOT_FULL_MARKET_PRICE_COLUMN)
map_boolean(VAT_EXCLUSIVE_COLUMN)


# 4. Handle Missing Values in Other Columns
# Fill missing Eircode with 'Unknown' (if column exists)
if EIRCODE_COLUMN in df.columns:
    df[EIRCODE_COLUMN].fillna('Unknown', inplace=True)
    print(f"Filled missing '{EIRCODE_COLUMN}'.")
else:
    print(f"Warning: Eircode column '{EIRCODE_COLUMN}' not found.")

# Fill missing Property Size Description with 'Unknown'
if PROPERTY_SIZE_COLUMN in df.columns:
    df[PROPERTY_SIZE_COLUMN].fillna('Unknown', inplace=True)
    print(f"Filled missing '{PROPERTY_SIZE_COLUMN}'.")
else:
    print(f"Warning: Property Size column '{PROPERTY_SIZE_COLUMN}' not found.")


# 5. Extract Features from Property Size Description (Improved Regex)
# This function is still limited but uses regex for slightly more flexibility.
# Consider treating PROPERTY_SIZE_COLUMN as categorical if this is too unreliable.
def extract_sqm_improved(description):
    if pd.isna(description) or description == 'Unknown':
        return np.nan
    
    description = description.lower()
    # Pattern for "greater than or equal to X less than Y sq metres"
    match = re.search(r'greater than or equal to (\d+)\s+less than (\d+)\s+sq metres', description)
    if match:
        return (int(match.group(1)) + int(match.group(2))) / 2
        
    # Pattern for "less than X sq metres"
    match = re.search(r'less than (\d+)\s+sq metres', description)
    if match:
        # Using the value itself as an upper bound estimate, could adjust (e.g., * 0.75)
        return int(match.group(1)) * 0.75 # Estimate below the threshold

    # Pattern for "greater than or equal to X sq metres"
    match = re.search(r'greater than or equal to (\d+)\s+sq metres', description)
    if match:
         # Using the value itself as a lower bound estimate, could adjust (e.g., * 1.25)
        return int(match.group(1)) * 1.1 # Estimate above the threshold
        
    # Pattern for just a number and "sq metres" (e.g., "approx 100 sq metres")
    match = re.search(r'(\d+)\s+sq metres', description)
    if match:
        return int(match.group(1))

    return np.nan # No recognized pattern

if PROPERTY_SIZE_COLUMN in df.columns:
    df['Approx_Size_SqM'] = df[PROPERTY_SIZE_COLUMN].apply(extract_sqm_improved)
    print(f"\nExtracted 'Approx_Size_SqM'. Missing values: {df['Approx_Size_SqM'].isnull().sum()}")
    # Fill remaining NaNs with the median *before* splitting
    size_median = df['Approx_Size_SqM'].median()
    df['Approx_Size_SqM'].fillna(size_median, inplace=True)
    print(f"Filled missing 'Approx_Size_SqM' with median: {size_median:.2f}")
else:
    df['Approx_Size_SqM'] = np.nan # Create column even if source is missing
    print("Warning: Could not create 'Approx_Size_SqM' as source column is missing.")


# 6. Feature Engineering: Temporal Features
df['Year'] = df[DATE_COLUMN].dt.year
df['Month'] = df[DATE_COLUMN].dt.month
df['Quarter'] = df[DATE_COLUMN].dt.quarter
df['DayOfWeek'] = df[DATE_COLUMN].dt.dayofweek # Monday=0, Sunday=6

# --- Define Features (X) and Target (y) ---
# Exclude target variable and columns directly derived from it if they were created
# Also exclude high-cardinality text fields not yet processed (like Address, Eircode unless encoded)
# Exclude original description columns if new features were derived
features_to_exclude = [
    TARGET_VARIABLE,
    DATE_COLUMN,
    # Add columns to exclude if they exist and shouldn't be features:
    'Address', # Usually too unique unless processed
    EIRCODE_COLUMN, # Too unique unless processed/binned
    PROPERTY_DESC_COLUMN, # Will be one-hot encoded instead
    PROPERTY_SIZE_COLUMN, # Replaced by Approx_Size_SqM or should be encoded
    price_col_actual # Drop original price column if different from TARGET_VARIABLE
]
# Ensure we only try to drop columns that actually exist
features_to_exclude = [col for col in features_to_exclude if col in df.columns]

# Define potential numerical and categorical features
potential_features = df.drop(columns=features_to_exclude).columns.tolist()

numerical_features = df[potential_features].select_dtypes(include=np.number).columns.tolist()
categorical_features = df[potential_features].select_dtypes(exclude=np.number).columns.tolist()

# Ensure boolean columns are treated correctly (sometimes detected as object/category)
bool_cols = [col for col in [NOT_FULL_MARKET_PRICE_COLUMN, VAT_EXCLUSIVE_COLUMN] if col in df.columns]
for b_col in bool_cols:
    if b_col in categorical_features:
        categorical_features.remove(b_col)
    if b_col not in numerical_features: # Add if not already numerical (though mapping should make them bool->int)
         numerical_features.append(b_col) # Treat bools as numerical (0/1) for scaling/modeling


# Add the property description column back for one-hot encoding
if PROPERTY_DESC_COLUMN in df.columns and PROPERTY_DESC_COLUMN not in categorical_features:
    categorical_features.append(PROPERTY_DESC_COLUMN)

print("\nSelected Numerical Features:", numerical_features)
print("Selected Categorical Features:", categorical_features)

# --- Prepare Final DataFrame for Splitting ---
# Include target variable for splitting, then separate X and y
cols_for_split = numerical_features + categorical_features + [TARGET_VARIABLE, DATE_COLUMN]
# Ensure columns exist before selecting
cols_for_split = [col for col in cols_for_split if col in df.columns]
final_df = df[cols_for_split].copy()

# --- Chronological Train-Test Split ---
final_df_sorted = final_df.sort_values(DATE_COLUMN)
train_size = int(len(final_df_sorted) * 0.8)

train_df = final_df_sorted.iloc[:train_size].drop(columns=[DATE_COLUMN])
test_df = final_df_sorted.iloc[train_size:].drop(columns=[DATE_COLUMN])

print(f"\nTrain set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# --- Preprocessing (Scaling and Encoding) - Fit on Train, Transform Train & Test ---

# Separate X and y for train and test sets
X_train = train_df.drop(columns=[TARGET_VARIABLE])
y_train = train_df[TARGET_VARIABLE]
X_test = test_df.drop(columns=[TARGET_VARIABLE])
y_test = test_df[TARGET_VARIABLE]

# Identify numerical and categorical columns within X_train
numerical_features_train = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_train = X_train.select_dtypes(exclude=np.number).columns.tolist()
# Add Property Desc back if it exists
if PROPERTY_DESC_COLUMN in X_train.columns and PROPERTY_DESC_COLUMN not in categorical_features_train:
     categorical_features_train.append(PROPERTY_DESC_COLUMN)


# 1. Scaling Numerical Features
scaler = StandardScaler()
# Fit *only* on training data
X_train[numerical_features_train] = scaler.fit_transform(X_train[numerical_features_train])
# Transform test data using the *same* scaler fitted on train data
X_test[numerical_features_train] = scaler.transform(X_test[numerical_features_train])
print("\nNumerical features scaled.")

# 2. Encoding Categorical Features (One-Hot Encoding)
# Use handle_unknown='ignore' to prevent errors if test set has categories not seen in train set
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Output dense array

# Fit *only* on training data
encoder.fit(X_train[categorical_features_train])

# Get feature names for new columns
encoded_feature_names = encoder.get_feature_names_out(categorical_features_train)

# Transform train data
X_train_encoded = encoder.transform(X_train[categorical_features_train])
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names, index=X_train.index)

# Transform test data
X_test_encoded = encoder.transform(X_test[categorical_features_train])
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names, index=X_test.index)

# Drop original categorical columns and concatenate encoded ones
X_train = X_train.drop(columns=categorical_features_train)
X_train = pd.concat([X_train, X_train_encoded_df], axis=1)

X_test = X_test.drop(columns=categorical_features_train)
X_test = pd.concat([X_test, X_test_encoded_df], axis=1)

print("Categorical features one-hot encoded.")
print(f"\nFinal X_train shape: {X_train.shape}")
print(f"Final X_test shape: {X_test.shape}")


# --- Save Processed Data ---
# Save X and y components separately for easy loading in modeling script
X_train.to_csv('property_X_train.csv', index=False)
y_train.to_csv('property_y_train.csv', index=False, header=True) # Include header for Series
X_test.to_csv('property_X_test.csv', index=False)
y_test.to_csv('property_y_test.csv', index=False, header=True) # Include header for Series

# Optional: Save the scaler and encoder for later use (e.g., on new data)
import joblib
joblib.dump(scaler, 'property_scaler.joblib')
joblib.dump(encoder, 'property_encoder.joblib')

print("\nProcessed data saved: property_X_train.csv, property_y_train.csv, property_X_test.csv, property_y_test.csv")
print("Scaler and Encoder saved: property_scaler.joblib, property_encoder.joblib")


# --- Exploratory Data Analysis (using the cleaned 'df' before splitting/scaling) ---
print("\nStarting Exploratory Data Analysis (EDA)...")
# Use the cleaned dataframe 'df' for EDA before scaling/encoding messes up distributions for plotting

if len(df) > 10: # Require a few data points for plots
    plt.figure(figsize=(12, 7))
    sns.histplot(df[TARGET_VARIABLE], kde=True, bins=50) # More bins might be useful
    plt.title(f'Distribution of Property Prices ({TARGET_VARIABLE})')
    plt.xlabel('Price (€)')
    plt.ylabel('Frequency')
    # Consider adding log scale for x-axis if heavily skewed
    # plt.xscale('log')
    plt.tight_layout()
    plt.savefig('price_distribution.png')
    print("Saved price_distribution.png")
    plt.close() # Close plot to free memory

    if PROPERTY_DESC_COLUMN in df.columns:
        plt.figure(figsize=(14, 8))
        # Calculate median price as it's less sensitive to outliers than mean
        property_prices = df.groupby(PROPERTY_DESC_COLUMN)[TARGET_VARIABLE].median().sort_values(ascending=False)
        property_prices.plot(kind='bar')
        plt.title(f'Median Property Price by Type ({PROPERTY_DESC_COLUMN})')
        plt.xlabel('Property Type')
        plt.ylabel('Median Price (€)')
        plt.xticks(rotation=60, ha='right') # Improve label readability
        plt.tight_layout()
        plt.savefig('property_type_prices.png')
        print("Saved property_type_prices.png")
        plt.close()

        plt.figure(figsize=(12, 7))
        df[PROPERTY_DESC_COLUMN].value_counts().plot(kind='bar')
        plt.title(f'Distribution of Property Types ({PROPERTY_DESC_COLUMN})')
        plt.xlabel('Property Type')
        plt.ylabel('Count')
        plt.xticks(rotation=60, ha='right')
        plt.tight_layout()
        plt.savefig('property_types.png')
        print("Saved property_types.png")
        plt.close()

    # Price trend over time
    year_counts = df['Year'].value_counts()
    if len(year_counts) > 1:
        plt.figure(figsize=(12, 6))
        # Use median price trend as well/instead
        time_series_median = df.groupby('Year')[TARGET_VARIABLE].median()
        time_series_median.plot(marker='o', label='Median Price')
        # Optional: Add mean price trend for comparison
        # time_series_mean = df.groupby('Year')[TARGET_VARIABLE].mean()
        # time_series_mean.plot(marker='x', linestyle='--', label='Mean Price')
        plt.title('Property Price Trend Over Time (Median)')
        plt.xlabel('Year')
        plt.ylabel('Median Price (€)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('price_trend.png')
        print("Saved price_trend.png")
        plt.close()

    print("\nEDA complete! Check the generated plots.")
else:
    print("\nNot enough data for meaningful EDA plots. Skipping visualization.")

