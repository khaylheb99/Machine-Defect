import streamlit as st
import subprocess
import sys


def install_package(package):
    try:
        __import__(package)
    except ImportError:
        st.warning(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install_package("joblib")

import pandas as pd
import joblib
# import scikit-learn==1.5.1 as sklearn

pipeline = joblib.load(r"C:\Users\Oluyemi Balogun\Downloads\Maintenance\pipeline.pkl")

def prediction(ProductionVolume, ProductionCost, SupplierQuality,
       DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
       DowntimePercentage, InventoryTurnover, StockoutRate,
       WorkerProductivity, SafetyIncidents, EnergyConsumption,
       EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost):
      
    input_data = {
      'ProductionVolume' : ProductionVolume,
      'ProductionCost' : ProductionCost,
      'SupplierQuality' : SupplierQuality,
      'DeliveryDelay' : DeliveryDelay,
      'DefectRate' : DefectRate,
      'QualityScore' : QualityScore,
      'MaintenanceHours' : MaintenanceHours,
      'DowntimePercentage' : DowntimePercentage,
      'InventoryTurnover' : InventoryTurnover,
      'StockoutRate' : StockoutRate,
      'WorkerProductivity' : WorkerProductivity,
      'SafetyIncidents' : SafetyIncidents,
      'EnergyConsumption' : EnergyConsumption,
      'EnergyEfficiency' : EnergyEfficiency,
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

try:
    import joblib
except ImportError:
    st.warning("Installing joblib...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.3.2"])
    import joblib

try:
    pipeline = joblib.load('pipeline.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error: {e}")


st.set_page_config(page_title="Manufacturing Defect Prediction", page_icon="🏭", layout="wide")
st.title("🏭 Manufacturing Defect Prediction App")

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


if st.button("🔍 Predict Defect Status"):
    result = prediction(
        ProductionVolume, ProductionCost, SupplierQuality,
        DeliveryDelay, DefectRate, QualityScore, MaintenanceHours,
        DowntimePercentage, InventoryTurnover, StockoutRate,
        WorkerProductivity, SafetyIncidents, EnergyConsumption,
        EnergyEfficiency, AdditiveProcessTime, AdditiveMaterialCost
    )
    st.success(result)

# import streamlit as st

# st.set_page_config(page_title="Test App", page_icon="✅", layout="wide")

# st.title("Hello Streamlit 👋")
# st.write("If you see this, Streamlit UI works fine.")

# try:
#     pipeline = joblib.load(r"C:\Users\Oluyemi Balogun\Downloads\Maintenance\pipeline.pkl")
# except Exception as e:
#     import traceback
#     print("❌ Error while loading pipeline:")
#     traceback.print_exc()

#     pipeline = None


