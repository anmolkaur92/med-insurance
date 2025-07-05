import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Try to import scikit-learn and check version
try:
    import sklearn
    from sklearn.preprocessing import StandardScaler
    st.write("✅ scikit-learn version:", sklearn.__version__)
except ModuleNotFoundError:
    st.error("❌ scikit-learn is not installed in this environment.")
except ImportError as e:
    st.error(f"❌ Import error: {e}")

# Load and re-save the model to ensure compatibility
try:
    


    import pickle

    print("🔄 Loading model with NumPy 2.x...")
    with open('MIPML.pkl', 'rb') as f:
        model = pickle.load(f)

    print("✅ Re-saving model with NumPy 1.x...")
    with open('MIPML_fixed.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("🎉 Model re-saved successfully!")



    # Check if the model is a valid predictive model
    if not hasattr(model, 'predict'):
        raise ValueError("The loaded model does not have a 'predict' method. Please check the model.")

except Exception as e:
    st.error(f"Error loading or re-saving model: {e}")
    raise

# App Title
st.header('Medical Insurance Premium Predictor')

# Taking user inputs
gender = st.selectbox('Choose Gender', ['Female', 'Male'])
smoker = st.selectbox('Are you a smoker?', ['Yes', 'No'])
region = st.selectbox('Choose Region', ['SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'])
age = st.slider('Enter Age', 5, 80)
bmi = st.slider('Enter BMI', 5, 100)
children = st.slider('Choose Number of Children', 0, 5)

# Predict button
if st.button('Predict'):
    # Encoding inputs manually
    gender_encoded = 0 if gender == 'Female' else 1
    smoker_encoded = 1 if smoker == 'Yes' else 0

    if region == 'SouthEast':
        region_encoded = 0
    elif region == 'SouthWest':
        region_encoded = 1
    elif region == 'NorthEast':
        region_encoded = 2
    else:
        region_encoded = 3  # NorthWest

    # Create input array
    input_data = (age, gender_encoded, bmi, children, smoker_encoded, region_encoded)
    input_data_array = np.asarray(input_data).reshape(1, -1)

    # Scaling input data (if required by model)
    try:
        scaler = StandardScaler()
        # NOTE: Ideally, you should load the scaler used during model training
        # Here, we fit it on the single input for demonstration only
        input_data_scaled = scaler.fit_transform(input_data_array)
    except Exception as e:
        st.error(f"Error in scaling the input data: {e}")
        raise

    # Make prediction
    try:
        predicted_prem = model.predict(input_data_scaled)
        if predicted_prem[0] < 0:
            st.error("Predicted premium is negative, which is not valid. Please check the model.")
        else:
            st.success(f'Insurance Premium will be {round(predicted_prem[0], 2)} rupees')
    except Exception as e:
        st.error(f"Error in prediction: {e}")
