import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="AgriAssist", page_icon="🌱", layout="wide")

# Load datasets (included in repo)
@st.cache_data
def load_data():
    crop_data = pd.read_csv('C:/Users/sukhm/OneDrive/Desktop/Agri/Agri/crop_data.csv')
    fertilizer_data = pd.read_csv('C:/Users/sukhm/OneDrive/Desktop/Agri/Agri/fertilizer.csv')
    return crop_data, fertilizer_data

crop_data, fertilizer_data = load_data()

# Ideal NPK values for crops (sample data - should be replaced with actual values)
ideal_values = {
    'rice': {'N': 120, 'P': 60, 'K': 60},
    'maize': {'N': 80, 'P': 40, 'K': 40},
    'chickpea': {'N': 40, 'P': 50, 'K': 60},
    'kidneybeans': {'N': 40, 'P': 50, 'K': 60},
    'pigeonpeas': {'N': 40, 'P': 50, 'K': 60},
    'mothbeans': {'N': 40, 'P': 50, 'K': 60},
    'mungbean': {'N': 40, 'P': 50, 'K': 60},
    'blackgram': {'N': 40, 'P': 50, 'K': 60},
    'lentil': {'N': 40, 'P': 50, 'K': 60},
    'pomegranate': {'N': 70, 'P': 50, 'K': 70},
    'banana': {'N': 100, 'P': 50, 'K': 100},
    'mango': {'N': 70, 'P': 50, 'K': 70},
    'grapes': {'N': 80, 'P': 50, 'K': 80},
    'watermelon': {'N': 60, 'P': 50, 'K': 80},
    'muskmelon': {'N': 60, 'P': 50, 'K': 80},
    'apple': {'N': 70, 'P': 50, 'K': 70},
    'orange': {'N': 70, 'P': 50, 'K': 70},
    'papaya': {'N': 80, 'P': 50, 'K': 80},
    'coconut': {'N': 70, 'P': 50, 'K': 70},
    'cotton': {'N': 90, 'P': 50, 'K': 70},
    'jute': {'N': 60, 'P': 50, 'K': 60},
    'coffee': {'N': 70, 'P': 50, 'K': 80}
}

# Train or load crop recommendation model
def get_crop_model():
    try:
        model = joblib.load('crop_model.pkl')
    except:
        # Load data
        crop_data = pd.read_csv('data/crop_data.csv')
        
        # Check if dataset is balanced
        st.write("Crop Distribution in Dataset:")
        st.bar_chart(crop_data['label'].value_counts())
        
        # Features & Target
        X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = crop_data['label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model trained! Accuracy: {accuracy:.2f}")
        
        # Save model
        joblib.dump(model, 'crop_model.pkl')
    
    return model

crop_model = get_crop_model()

# Fertilizer recommendation function
def recommend_fertilizer(crop, N, P, K):
    try:
        ideal = ideal_values[crop.lower()]
    except KeyError:
        return "Crop not found in database", None
    
    deficiencies = []
    
    # Calculate deficiencies
    if N < ideal['N']:
        deficiencies.append(('Nitrogen', ideal['N'] - N, 'Urea'))
    if P < ideal['P']:
        deficiencies.append(('Phosphorus', ideal['P'] - P, 'DAP (Diammonium Phosphate)'))
    if K < ideal['K']:
        deficiencies.append(('Potassium', ideal['K'] - K, 'MOP (Muriate of Potash)'))
    
    if not deficiencies:
        return "Your soil has sufficient nutrients for this crop. No fertilizer needed.", None
    
    # Find the most deficient nutrient
    deficiencies.sort(key=lambda x: x[1], reverse=True)
    main_def = deficiencies[0]
    
    # Calculate dosage (simplified calculation)
    dosage = round(main_def[1] * 1.5)  # kg per acre
    
    # Prepare alternative recommendations
    alternatives = []
    if main_def[0] == 'Nitrogen':
        alternatives = ['Compost manure', 'Fish emulsion', 'Blood meal']
    elif main_def[0] == 'Phosphorus':
        alternatives = ['Bone meal', 'Rock phosphate', 'Compost']
    else:
        alternatives = ['Wood ash', 'Greensand', 'Compost']
    
    return main_def, deficiencies, dosage, alternatives

# Streamlit UI
def main():
    st.title("🌱 AgriAssist: Smart Farming Assistant")
    st.markdown("""
    A recommendation system to help farmers choose the best crops and fertilizers for their land.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Crop Recommendation", "Fertilizer Recommendation", "About"])
    
    with tab1:
        st.header("Crop Recommendation")
        st.write("Enter your soil and climate conditions to get crop recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            N = st.slider("Nitrogen (N) level", 0, 150, 50)
            P = st.slider("Phosphorus (P) level", 0, 150, 50)
            K = st.slider("Potassium (K) level", 0, 150, 50)
            ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
        
        with col2:
            rainfall = st.slider("Rainfall (mm)", 0, 500, 100)
            temperature = st.slider("Temperature (°C)", 0, 50, 25)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
        
        if st.button("Get Crop Recommendations"):
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_model.predict(input_data)
            probabilities = crop_model.predict_proba(input_data)[0]
            
            st.success(f"Recommended crop: **{prediction[0].capitalize()}**")
            
            # Show top 5 crops
            crops = crop_model.classes_
            top5_idx = np.argsort(probabilities)[-5:][::-1]
            
            st.subheader("Top 5 Suitable Crops")
            fig, ax = plt.subplots()
            sns.barplot(x=probabilities[top5_idx], y=[crops[i].capitalize() for i in top5_idx], ax=ax)
            ax.set_xlabel("Probability")
            ax.set_ylabel("Crop")
            st.pyplot(fig)
    
    with tab2:
        st.header("Fertilizer Recommendation")
        st.write("Get fertilizer recommendations based on your crop and soil nutrient levels")
        
        crop = st.selectbox("Select your crop", list(ideal_values.keys()))
        col1, col2, col3 = st.columns(3)
        
        with col1:
            N = st.slider("Current Nitrogen (N) level", 0, 150, 30, key='fert_N')
        with col2:
            P = st.slider("Current Phosphorus (P) level", 0, 150, 20, key='fert_P')
        with col3:
            K = st.slider("Current Potassium (K) level", 0, 150, 25, key='fert_K')
        
        if st.button("Get Fertilizer Recommendation"):
            main_def, all_def, dosage, alternatives = recommend_fertilizer(crop, N, P, K)
            
            if isinstance(main_def, str):
                st.success(main_def)
            else:
                st.warning(f"Your soil lacks **{main_def[0]}**. The deficiency is **{main_def[1]:.1f} kg/acre**")
                st.info(f"**Recommendation:** Apply **{dosage} kg/acre** of **{main_def[2]}**")
                
                st.subheader("All Nutrient Deficiencies")
                def_df = pd.DataFrame(all_def, columns=['Nutrient', 'Deficiency (kg/acre)', 'Recommended Fertilizer'])
                st.dataframe(def_df.style.highlight_max(axis=0, subset=['Deficiency (kg/acre)']))
                
                st.subheader("Organic Alternatives")
                for alt in alternatives:
                    st.write(f"- {alt}")
                
                # Visualize nutrient levels
                ideal = ideal_values[crop.lower()]
                nutrients = ['Nitrogen', 'Phosphorus', 'Potassium']
                current = [N, P, K]
                ideal_vals = [ideal['N'], ideal['P'], ideal['K']]
                
                fig, ax = plt.subplots()
                x = range(len(nutrients))
                width = 0.35
                
                ax.bar(x, current, width, label='Current')
                ax.bar([i + width for i in x], ideal_vals, width, label='Ideal')
                ax.set_xticks([i + width/2 for i in x])
                ax.set_xticklabels(nutrients)
                ax.set_ylabel('Nutrient Level')
                ax.set_title('Current vs Ideal Nutrient Levels')
                ax.legend()
                
                st.pyplot(fig)
    
    with tab3:
        st.header("About AgriAssist")
        st.markdown("""
        AgriAssist is an intelligent farming assistant that provides:
        
        - **Crop recommendations** based on soil and climate conditions
        - **Fertilizer suggestions** based on nutrient deficiencies
        - Data-driven insights to help farmers maximize yield
        
        ### How It Works
        1. The system uses machine learning models trained on agricultural data
        2. For crop recommendation, it considers 7 parameters:
           - N, P, K levels
           - Temperature, humidity
           - pH, rainfall
        3. For fertilizer recommendation, it compares current soil nutrients with ideal values
        
        ### Future Enhancements
        - Yield prediction
        - Pest/disease alerts
        - Market price forecasts
        - Multi-language support
        """)
        
        st.subheader("Sample Data")
        st.write("Crop Recommendation Dataset:")
        st.dataframe(crop_data.head())
        st.write("Fertilizer Recommendation Data:")
        st.dataframe(fertilizer_data.head())

if __name__ == "__main__":
    main()