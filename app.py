import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Manufacturing Defect Prediction", page_icon="🏭", layout="wide")

@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('pipeline.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model()

def prediction(ProductionVolume, ProductionCost, SupplierQuality,
       DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
       DowntimePercentage, InventoryTurnover, StockoutRate,
       WorkerProductivity, SafetyIncidents, EnergyConsumption,
       EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost):
      
    input_data = {
      'ProductionVolume': ProductionVolume,
      'ProductionCost': ProductionCost,
      'SupplierQuality': SupplierQuality,
      'DeliveryDelay': DeliveryDelay,
      'DefectRate': DefectRate,
      'QualityScore': QualityScore,
      'MaintenanceHours': MaintenanceHours,
      'DowntimePercentage': DowntimePercentage,
      'InventoryTurnover': InventoryTurnover,
      'StockoutRate': StockoutRate,
      'WorkerProductivity': WorkerProductivity,
      'SafetyIncidents': SafetyIncidents,
      'EnergyConsumption': EnergyConsumption,
      'EnergyEfficiency': EnergyEfficiency,
      'AdditiveProcessTime': AdditiveProcessTime,
      'AdditiveMaterialCost': AdditiveMaterialCost
    }
    
    input_df = pd.DataFrame([input_data])
    
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df).max()
    
    if pred == 1:
        return f"No Defect: Certainty {prob*100:.2f}%"
    else:
        return f"Defect: Certainty {prob*100:.2f}%"


# Main app interface
st.title("🏭 Manufacturing Defect Prediction App")

if pipeline is None:
    st.error("""
    **Model failed to load. Please check:**
    1. requirements.txt is correct and in repository root
    2. pipeline.pkl exists in repository
    3. All files are committed to git
    """)
else:
    st.success("Model loaded successfully!")
    
    col1, col2 = st.columns(2)

    with col1:
        ProductionVolume = st.number_input("Production Volume", min_value=0.0, value=202.0)
        ProductionCost = st.number_input("Production Cost", min_value=0.0, value=13175.40)
        SupplierQuality = st.slider("Supplier Quality", 0.0, 100.0, 86.65)
        DeliveryDelay = st.number_input("Delivery Delay (days)", min_value=0.0, value=1.0)
        DefectRate = st.number_input("Defect Rate", min_value=0.0, value=3.12)
        QualityScore = st.slider("Quality Score", 0.0, 100.0, 63.46)
        MaintenanceHours = st.number_input("Maintenance Hours", min_value=0.0, value=9.0)
        DowntimePercentage = st.number_input("Downtime %", min_value=0.0, value=0.052343)

    with col2:
        InventoryTurnover = st.number_input("Inventory Turnover", min_value=0.0, value=8.63)
        StockoutRate = st.number_input("Stockout Rate", min_value=0.0, value=0.081)
        WorkerProductivity = st.slider("Worker Productivity", 0.0, 100.0, 85.04)
        SafetyIncidents = st.number_input("Safety Incidents", min_value=0, value=0)
        EnergyConsumption = st.number_input("Energy Consumption", min_value=0.0, value=2419.62)
        EnergyEfficiency = st.slider("Energy Efficiency", 0.0, 1.0, 0.4689)
        AdditiveProcessTime = st.number_input("Additive Process Time", min_value=0.0, value=5.55)
        AdditiveMaterialCost = st.number_input("Additive Material Cost", min_value=0.0, value=236.44)

    if st.button("Predict Defect Status"):
        with st.spinner("Analyzing..."):
            result = prediction(
                ProductionVolume, ProductionCost, SupplierQuality,
                DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
                DowntimePercentage, InventoryTurnover, StockoutRate,
                WorkerProductivity, SafetyIncidents, EnergyConsumption,
                EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost
            )
        st.success(result)
