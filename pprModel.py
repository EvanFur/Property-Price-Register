import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re
import joblib


# Load the data - using the specific encoding you have in your script
df = pd.read_csv('PPR-ALL.csv', encoding='Windows-1252')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing functions
def clean_price(price_str):
    """Convert price from string format to float"""
    if pd.isna(price_str):
        return np.nan
    # Remove euro symbol and commas, then convert to float
    return float(re.sub(r'[€,]', '', price_str))

def extract_county_from_address(address):
    """Extract county from address if possible"""
    if pd.isna(address):
        return np.nan
    
    # Check if "Co." is in the address
    match = re.search(r'Co\.\s*([A-Za-z]+)', address)
    if match:
        return match.group(1)
    
    # Check for Dublin patterns
    dublin_match = re.search(r'Dublin\s*\d*', address)
    if dublin_match:
        return "Dublin"
    
    return np.nan

def extract_features(df):
    """Extract and create features from the dataset"""
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Process date - FIXED: First check if column exists
    date_col = 'Date of Sale (dd/mm/yyyy)'
    if date_col in df_processed.columns:
        # Convert to datetime safely
        try:
            df_processed['Sale_Date'] = pd.to_datetime(df_processed[date_col], format='%d/%m/%Y', errors='coerce')
            df_processed['Sale_Year'] = df_processed['Sale_Date'].dt.year
            df_processed['Sale_Month'] = df_processed['Sale_Date'].dt.month
        except Exception as e:
            print(f"Error converting dates: {e}")
            # Create default values if conversion fails
            df_processed['Sale_Year'] = 0
            df_processed['Sale_Month'] = 0
    else:
        print(f"Warning: '{date_col}' column not found")
        df_processed['Sale_Year'] = 0
        df_processed['Sale_Month'] = 0
    
    # Process price
    price_col = 'Price (€)'
    if price_col in df_processed.columns:
        df_processed['Price_Numeric'] = df_processed[price_col].apply(clean_price)
    else:
        print(f"Warning: '{price_col}' column not found")
        df_processed['Price_Numeric'] = 0
    
    # Extract county if missing
    df_processed['County_Filled'] = df_processed['County'] if 'County' in df_processed.columns else pd.Series([np.nan] * len(df_processed))
    missing_counties = df_processed['County_Filled'].isna()
    
    if 'Address' in df_processed.columns:
        df_processed.loc[missing_counties, 'County_Filled'] = df_processed.loc[missing_counties, 'Address'].apply(extract_county_from_address)
    
    # Create binary features
    desc_col = 'Description of Property'
    if desc_col in df_processed.columns:
        df_processed['Is_New'] = (df_processed[desc_col] == 'New Dwelling house /Apartment').astype(int)
        df_processed['Is_Second_Hand'] = (df_processed[desc_col] == 'Second-Hand Dwelling house /Apartment').astype(int)
    else:
        print(f"Warning: '{desc_col}' column not found")
        df_processed['Is_New'] = 0
        df_processed['Is_Second_Hand'] = 0
    
    # Process size information
    size_col = 'Property Size Description'
    if size_col in df_processed.columns:
        df_processed['Has_Size_Info'] = (~df_processed[size_col].isna()).astype(int)
        df_processed['Is_Small'] = df_processed[size_col].str.contains('less than', case=False, na=False).astype(int)
        df_processed['Is_Medium'] = df_processed[size_col].str.contains('greater than or equal to 38 sq metres and less than 125 sq metres', case=False, na=False).astype(int)
        df_processed['Is_Large'] = df_processed[size_col].str.contains('greater than', case=False, na=False).astype(int) & (~df_processed['Is_Medium'].astype(bool)).astype(int)
    else:
        print(f"Warning: '{size_col}' column not found")
        df_processed['Has_Size_Info'] = 0
        df_processed['Is_Small'] = 0
        df_processed['Is_Medium'] = 0
        df_processed['Is_Large'] = 0
    
    # Binary flags
    flag_col1 = 'Not Full Market Price'
    if flag_col1 in df_processed.columns:
        df_processed['Not_Full_Market_Price'] = (df_processed[flag_col1] == 'Yes').astype(int)
    else:
        print(f"Warning: '{flag_col1}' column not found")
        df_processed['Not_Full_Market_Price'] = 0
        
    flag_col2 = 'VAT Exclusive'
    if flag_col2 in df_processed.columns:
        df_processed['VAT_Exclusive'] = (df_processed[flag_col2] == 'Yes').astype(int)
    else:
        print(f"Warning: '{flag_col2}' column not found")
        df_processed['VAT_Exclusive'] = 0
    
    return df_processed

# Process the data
print("Processing data...")
processed_df = extract_features(df)

# Print processed data info
print("\nProcessed data columns:")
print(processed_df.columns.tolist())

# Prepare features and target
# We'll use a subset of features for this example
features = ['Sale_Year', 'Sale_Month', 'County_Filled', 'Is_New', 'Is_Second_Hand', 
            'Has_Size_Info', 'Is_Small', 'Is_Medium', 'Is_Large', 
            'Not_Full_Market_Price', 'VAT_Exclusive']

# Check if we have price data to train the model
if 'Price_Numeric' not in processed_df.columns or processed_df['Price_Numeric'].isna().all():
    print("Error: No valid price data found for training")
    exit(1)

# Remove rows with missing target
valid_data = processed_df.dropna(subset=['Price_Numeric'])
print(f"Valid data for training: {len(valid_data)} rows")

if len(valid_data) == 0:
    print("Error: No valid data for training after removing NaN values")
    exit(1)

X = valid_data[features]
y = valid_data['Price_Numeric']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessor
categorical_features = ['County_Filled']
numerical_features = [col for col in features if col not in categorical_features]

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train model
print("Training model...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: €{mae:.2f}")
print(f"Root Mean Squared Error: €{rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
if hasattr(model[-1], 'feature_importances_'):
    # Get feature names safely
    try:
        feature_names = (
            numerical_features + 
            model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features).tolist()
        )
        importances = model[-1].feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved as 'feature_importance.png'")
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

# Example prediction function
def predict_price(model, county, year, month, is_new=0, is_second_hand=1, 
                  has_size_info=1, is_small=0, is_medium=1, is_large=0,
                  not_full_price=0, vat_exclusive=0):
    """Make a prediction using the trained model"""
    data = {
        'Sale_Year': [year],
        'Sale_Month': [month],
        'County_Filled': [county],
        'Is_New': [is_new],
        'Is_Second_Hand': [is_second_hand],
        'Has_Size_Info': [has_size_info],
        'Is_Small': [is_small],
        'Is_Medium': [is_medium],
        'Is_Large': [is_large],
        'Not_Full_Market_Price': [not_full_price],
        'VAT_Exclusive': [vat_exclusive]
    }
    input_df = pd.DataFrame(data)
    prediction = model.predict(input_df)[0]
    return prediction

# Example predictions
print("\nExample predictions:")
print(f"Dublin, 2023, second-hand, medium size: €{predict_price(model, 'Dublin', 2023, 6):.2f}")
print(f"Meath, 2023, new property, large size: €{predict_price(model, 'Meath', 2023, 6, is_new=1, is_second_hand=0, is_medium=0, is_large=1):.2f}")

# Save the trained model to a file
joblib.dump(model, 'irish_property_price_model.pkl')
print("Model saved to 'irish_property_price_model.pkl'")