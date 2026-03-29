# 🌊 FedFlood — Flood Forecasting Model Using Federated Learning

A privacy-preserving flood forecasting dashboard built with **Streamlit**, **TensorFlow**, and **Plotly**. The system uses Federated Learning (FedAvg) to train a Feed-Forward Neural Network across 18 distributed client stations without sharing raw sensor data. Trained on a **50,000-row real-world Indian flood risk dataset**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project implements the **FedFlood** algorithm, a federated learning-based approach to flood prediction that enables multiple monitoring stations to collaboratively train a shared model while keeping their data local. The model is trained on a real-world Indian flood risk dataset containing 50,000 records across 14 hydrological and geographical features.

### Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.2%** |
| **RMSE** | **0.1665** |
| **R² Score** | **0.8817** |
| Forecast Lead Time | 5 days |
| Training Samples | 40,000 (80% split) |
| Test Samples | 10,000 (20% split) |

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      FedFlood Dashboard                       │
├──────────────┬───────────────────┬────────────────────────────┤
│ 📡 Station   │ ⚡ Risk            │ 🔗 FL                      │
│    Monitor   │    Predictor      │    Simulator                │
├──────────────┴───────────────────┴────────────────────────────┤
│          TensorFlow FFNN (11→128→64→32→16→1)                  │
│     BatchNormalization + Dropout(0.3) after each layer         │
├───────────────────────────────────────────────────────────────┤
│     Weighted FedAvg — Federated Averaging (18 clients)        │
│     50,000-row Indian Flood Risk Dataset                      │
└───────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The model is trained on `flood_risk_dataset_india.csv` — a **50,000-row** dataset with 14 columns capturing hydrological, geographical, and environmental features across India.

### Features Used (11 after preprocessing)

| Feature | Type | Description |
|---------|------|-------------|
| Rainfall (mm) | Numeric | Precipitation amount |
| Temperature (°C) | Numeric | Ambient temperature |
| Humidity (%) | Numeric | Relative humidity |
| River Discharge (m³/s) | Numeric | River flow rate |
| Water Level (m) | Numeric | Current water level |
| Elevation (m) | Numeric | Terrain elevation |
| Land Cover | Categorical | Forest, Agricultural, Urban, etc. |
| Soil Type | Categorical | Clay, Silt, Loam, Sandy, etc. |
| Population Density | Numeric | People per unit area |
| Infrastructure | Numeric | Infrastructure index |
| Historical Floods | Numeric | Past flood event count |

### Dropped Features
- **Latitude** and **Longitude** — removed as they are not directly predictive for flood occurrence.

### Target Variable
- **Flood Occurred** — Binary (0 = No Flood, 1 = Flood)
- Class distribution: 31,260 (No) / 18,740 (Yes)

### Preprocessing Pipeline
1. **LabelEncoder** on `Land Cover` and `Soil Type`
2. **StandardScaler** on all 11 features
3. **Stratified 80/20 split** (preserves class balance)

---

## 🖥️ Dashboard Pages

### 📡 Station Monitor
- Real-time monitoring of **18 client stations** across 5 Indian rivers (Ganga, Brahmaputra, Yamuna, Godavari, Mahanadi)
- Risk-level badges (Critical, High, Moderate, Low) with animated alerts
- 5-Day Water Level Forecast (interactive spline charts)
- Precipitation Efficiency Comparison

### ⚡ Risk Predictor
- **6 input sliders**: Rainfall, Water Level, Snow Melt, Upstream Inflow, Duration, Hydrodynamic Index
- Real-time flood probability gauge (0–100%)
- Model diagnostics: RMSE, R², Accuracy
- Sub-model contribution breakdown:
  - **Snow Melt** — Temperature + Humidity
  - **Rainfall-Runoff** — Rainfall + River Discharge
  - **Flow Routing** — Water Level + Elevation
  - **Hydrodynamics** — Land Cover + Soil Type + Population Density + Infrastructure + Historical Floods

### 🔗 FL Simulator
- Live **Weighted FedAvg** animation across 18 TensorFlow client models
- Configurable: training rounds (3–25), active clients (6–18), local epochs (1–5)
- Round-by-round loss chart + accuracy ring gauge
- Network transfer tracker (weights-only, privacy-preserving)

---

## 🔒 Privacy-Preserving Design

```
Station 1  ──┐                               ┌──  Station 1
Station 2  ──┤   weights only                ├──  Station 2
    ...      ├──────────────► Server ────────►┤       ...
Station 18 ──┘   (no raw data)               └──  Station 18
                   Weighted FedAvg
```

Raw sensor data **never leaves** the local station. Only model weights are exchanged with the aggregation server. Each client is weighted by its data shard size during aggregation.

---

## 🧠 Model Details

### FFNN Architecture (Upgraded)

```
Input (11 features)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(32, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(16, ReLU)  → BatchNorm → Dropout(0.3)
    ↓
Dense(1, Sigmoid) → Flood Probability (0–100%)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Crossentropy |
| Batch Size | 64 |
| Max Epochs | 150 |
| EarlyStopping | patience=20, restore_best_weights=True |
| ReduceLROnPlateau | patience=7, factor=0.3 |
| Validation | 20% stratified test set |

### Federated Learning (Weighted FedAvg)
1. Server broadcasts global model weights to all 18 clients
2. Real training data is split into **18 equal shards** (one per client)
3. Each client trains locally on its own data shard
4. Clients send only updated weights back to server
5. Server performs **weighted averaging** (weight = shard size / total samples)
6. Repeat for N rounds

---

## 📁 Project Structure

```
FFM-Flood-Forecasting-Model-Using-Federated-Learning/
├── app.py                          # Streamlit dashboard (3 pages + glassmorphism CSS)
├── models.py                       # TensorFlow FFNN + FedAvg + data preprocessing
├── stations.py                     # 18 Indian monitoring stations (CWC-modelled data)
├── flood_risk_dataset_india.csv    # 50,000-row Indian flood risk dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Key Functions in `models.py`

| Function | Description |
|----------|-------------|
| `_load_and_preprocess(csv_path)` | Load CSV, encode categoricals, scale features, stratified split |
| `build_ffnn(input_dim, seed)` | Build the FFNN with BatchNorm + Dropout layers |
| `train_global_model(csv_path)` | Full training pipeline → returns model, scaler, metrics, encoders, data |
| `predict_flood_risk(model, inputs, ...)` | Run inference and return probability, forecast, diagnostics |
| `federated_round(global_model, client_ids, ...)` | One round of weighted FedAvg across client shards |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/abhilash2272/FFM-Flood-Forecasting-Model-Using-Federated-Learning.git
cd FFM-Flood-Forecasting-Model-Using-Federated-Learning

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `tensorflow` | FFNN model + federated training |
| `plotly` | Interactive charts and gauges |
| `pandas` | CSV data loading and manipulation |
| `numpy` | Numerical computations |
| `scikit-learn` | Preprocessing (LabelEncoder, StandardScaler) + metrics |

---

## 🏞️ Monitored Rivers & Stations

| River | Stations | Count |
|-------|----------|-------|
| **Ganga** | Farakka Barrage, Haridwar, Varanasi, Patna, Prayagraj | 5 |
| **Brahmaputra** | Guwahati, Dibrugarh, Tezpur, Dhubri | 4 |
| **Yamuna** | Delhi (Old Rly Bridge), Mathura, Agra, Etawah | 4 |
| **Godavari** | Nashik, Rajahmundry, Polavaram | 3 |
| **Mahanadi** | Hirakud Dam, Naraj | 2 |

Station data is modelled on **CWC (Central Water Commission)** telemetry patterns.

---

## 🎨 UI Design

The dashboard features a **premium glassmorphism design** with:
- Dark theme with indigo/violet gradient accents
- Frosted glass cards with backdrop blur effects
- Animated risk badges and danger alerts
- Interactive Plotly charts with custom dark theme
- JetBrains Mono typography for data values
- Responsive layout with hover micro-animations

---
