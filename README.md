# 🌲 Forest Cover Type Prediction

Predict the type of forest cover using environmental and geographic features from the Roosevelt National Forest dataset.

---

## 📌 Objective
Build a machine learning model to classify forest cover types (1 to 7) based on terrain and remote sensing data from 30m × 30m patches of land.

---

## 📊 Dataset Overview

- **Source**: Roosevelt National Forest, Colorado
- **Target Variable**: `Cover_Type` (values 1 to 7)
- **Features**:
  - Elevation, Aspect, Slope
  - Distances to hydrology, roads, fire points
  - Hillshade (9AM, Noon, 3PM)
  - One-hot encoded soil types (40) & wilderness areas (4)

---

## 🧠 Model Used

- **Algorithm**: Random Forest Classifier  
- **Library**: scikit-learn  
- **Accuracy**: ~87.6%

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Andythakur/forest-cover-prediction.git
   cd forest-cover-prediction

pip install -r requirements.txt
python forest_cover_type_prediction.py

---

### 📦 `requirements.txt`

Save this as `requirements.txt`:
