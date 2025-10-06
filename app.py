import streamlit as st
import pandas as pd
import joblib

# App configuration
st.set_page_config(page_title="Manufacturing Defect Prediction", page_icon="🏭", layout="wide")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load('pipeline.pkl')
        st.success("Model loaded successfully!")
        return pipeline
    except FileNotFoundError:
        st.error("Model file 'pipeline.pkl' not found. Please make sure it's in your repository.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
pipeline = load_model()

def predict_defect(ProductionVolume, ProductionCost, SupplierQuality,
                   DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
                   DowntimePercentage, InventoryTurnover, StockoutRate,
                   WorkerProductivity, SafetyIncidents, EnergyConsumption,
                   EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost):
    
    input_data = {
        'ProductionVolume': [ProductionVolume],
        'ProductionCost': [ProductionCost],
        'SupplierQuality': [SupplierQuality],
        'DeliveryDelay': [DeliveryDelay],
        'DefectRate': [DefectRate],
        'QualityScore': [QualityScore],
        'MaintenanceHours': [MaintenanceHours],
        'DowntimePercentage': [DowntimePercentage],
        'InventoryTurnover': [InventoryTurnover],
        'StockoutRate': [StockoutRate],
        'WorkerProductivity': [WorkerProductivity],
        'SafetyIncidents': [SafetyIncidents],
        'EnergyConsumption': [EnergyConsumption],
        'EnergyEfficiency': [EnergyEfficiency],
        'AdditiveProcessTime': [AdditiveProcessTime],
        'AdditiveMaterialCost': [AdditiveMaterialCost]
    }
    
    input_df = pd.DataFrame(input_data)
    
    try:
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df).max()
        
        if prediction == 1:
            return f"No Defect Detected (Confidence: {probability*100:.2f}%)"
        else:
            return f"Defect Detected (Confidence: {probability*100:.2f}%)"
    except Exception as e:
        return f"Prediction error: {e}"

# Main app interface
st.title("🏭 Manufacturing Defect Prediction App")
st.markdown("Predict potential defects in manufacturing processes based on operational parameters")

if pipeline is not None:
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Production Metrics")
        ProductionVolume = st.number_input("Production Volume", min_value=0.0, value=202.0, help="Total units produced")
        ProductionCost = st.number_input("Production Cost ($)", min_value=0.0, value=13175.40, help="Total production cost")
        SupplierQuality = st.slider("Supplier Quality Score", 0.0, 100.0, 86.65, help="Quality rating from suppliers (0-100)")
        DeliveryDelay = st.number_input("Delivery Delay (days)", min_value=0.0, value=1.0, help="Average delivery delay in days")
        DefectRate = st.number_input("Defect Rate (%)", min_value=0.0, value=3.12, help="Current defect rate percentage")
        QualityScore = st.slider("Quality Score", 0.0, 100.0, 63.46, help="Overall quality assessment score")
        MaintenanceHours = st.number_input("Maintenance Hours", min_value=0.0, value=9.0, help="Hours spent on maintenance")
        DowntimePercentage = st.number_input("Downtime Percentage", min_value=0.0, max_value=100.0, value=0.052343, help="Percentage of downtime")

    with col2:
        st.subheader("Operational Metrics")
        InventoryTurnover = st.number_input("Inventory Turnover", min_value=0.0, value=8.63, help="Inventory turnover ratio")
        StockoutRate = st.number_input("Stockout Rate (%)", min_value=0.0, value=0.081, help="Rate of stock unavailability")
        WorkerProductivity = st.slider("Worker Productivity", 0.0, 100.0, 85.04, help="Worker productivity score")
        SafetyIncidents = st.number_input("Safety Incidents", min_value=0, value=0, help="Number of safety incidents reported")
        EnergyConsumption = st.number_input("Energy Consumption (kWh)", min_value=0.0, value=2419.62, help="Total energy consumption")
        EnergyEfficiency = st.slider("Energy Efficiency", 0.0, 1.0, 0.4689, help="Energy efficiency ratio (0-1)")
        AdditiveProcessTime = st.number_input("Additive Process Time (hrs)", min_value=0.0, value=5.55, help="Time for additive processes")
        AdditiveMaterialCost = st.number_input("Additive Material Cost ($)", min_value=0.0, value=236.44, help="Cost of additive materials")

    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 Predict Defect Status", type="primary", use_container_width=True):
        with st.spinner("Analyzing manufacturing parameters..."):
            result = predict_defect(
                ProductionVolume, ProductionCost, SupplierQuality,
                DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
                DowntimePercentage, InventoryTurnover, StockoutRate,
                WorkerProductivity, SafetyIncidents, EnergyConsumption,
                EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost
            )
        
        # Display result
        if "Defect Detected" in result:
            st.error(result)
        else:
            st.success(result)

else:
    st.warning("""
    **Setup Instructions:**
    1. Make sure `pipeline.pkl` is uploaded to your GitHub repository
    2. Verify `requirements.txt` contains all required packages
    3. All files should be in the root directory of your repository
    """)
