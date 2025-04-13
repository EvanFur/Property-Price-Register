import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    county = request.form.get('county')
    year = int(request.form.get('year'))
    month = int(request.form.get('month'))
    property_type = request.form.get('property_type')
    size = request.form.get('size')
    
    # Process inputs
    is_new = 1 if property_type == "new" else 0
    is_second_hand = 1 if property_type == "second-hand" else 0
    
    is_small = 1 if size == "small" else 0
    is_medium = 1 if size == "medium" else 0
    is_large = 1 if size == "large" else 0
    
    # Make prediction
    predicted_price = predict_price(
        model, county, year, month,
        is_new, is_second_hand,
        1, is_small, is_medium, is_large
    )
    
    return render_template('index.html', 
                          prediction=f"€{predicted_price:,.2f}", 
                          county=county,
                          property_type=property_type,
                          size=size)

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    
    # Process inputs
    county = data.get('county')
    year = int(data.get('year'))
    month = int(data.get('month'))
    property_type = data.get('property_type')
    size = data.get('size')
    
    is_new = 1 if property_type == "new" else 0
    is_second_hand = 1 if property_type == "second-hand" else 0
    
    is_small = 1 if size == "small" else 0
    is_medium = 1 if size == "medium" else 0
    is_large = 1 if size == "large" else 0
    
    # Make prediction
    predicted_price = predict_price(
        model, county, year, month,
        is_new, is_second_hand,
        1, is_small, is_medium, is_large
    )
    
    return jsonify({
        'predicted_price': predicted_price,
        'formatted_price': f"€{predicted_price:,.2f}",
        'property': {
            'county': county,
            'year': year,
            'month': month,
            'property_type': property_type,
            'size': size
        }
    })

if __name__ == '__main__':
    app.run(debug=True)