import joblib
import pandas as pd
import json
import sys

# Load the pre-trained model
model = joblib.load('irish_property_price_model.pkl')

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_property.py property_config.json")
        sys.exit(1)
        
    # Load property details from JSON file
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        property_details = json.load(f)
    
    # Process size
    size = property_details.get('size', 'medium')
    is_small = 1 if size == 'small' else 0
    is_medium = 1 if size == 'medium' else 0
    is_large = 1 if size == 'large' else 0
    
    # Make prediction
    predicted_price = predict_price(
        model, 
        property_details['county'], 
        property_details['year'], 
        property_details['month'],
        property_details.get('is_new', 0),
        property_details.get('is_second_hand', 1),
        1, is_small, is_medium, is_large
    )
    
    # Display result
    property_type = "new" if property_details.get('is_new', 0) == 1 else "second-hand"
    print(f"Predicted price for a {size} {property_type} property in {property_details['county']}: €{predicted_price:,.2f}")