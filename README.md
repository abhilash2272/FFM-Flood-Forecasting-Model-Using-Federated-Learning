# 🌊 FedFlood — Flood Forecasting Model Using Federated Learning

A privacy-preserving flood forecasting dashboard built with **Streamlit**, **TensorFlow**, and **Plotly**. The system uses Federated Learning (FedAvg) to train a Feed-Forward Neural Network across 18 distributed client stations without sharing raw sensor data.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project implements the **FedFlood** algorithm, a federated learning-based approach to flood prediction that enables multiple monitoring stations to collaboratively train a shared model while keeping their data local.

### Key Metrics (Paper Ground Truth)
| Metric | Value |
|--------|-------|
| Accuracy | **84%** |
| R² Score | **0.99** |
| RMSE | **0.2 – 0.5** |
| Forecast Lead Time | **5 days** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FedFlood Dashboard                  │
├──────────────┬──────────────────┬─────────────────────┤
│ 📡 Station   │ ⚡ Risk          │ 🔗 FL               │
│    Monitor   │    Predictor     │    Simulator         │
├──────────────┴──────────────────┴─────────────────────┤
│              TensorFlow FFNN (6→64→32→16→1)           │
├───────────────────────────────────────────────────────┤
│           FedAvg — Federated Averaging                │
│     (18 clients, weights-only aggregation)            │
└───────────────────────────────────────────────────────┘
```

---

## 🖥️ Dashboard Pages

### 📡 Station Monitor
- Real-time monitoring of **18 client stations** across 5 Indian rivers (Ganga, Brahmaputra, Yamuna, Godavari, Mahanadi)
- Risk-level badges (Critical, High, Moderate, Low) with animated alerts
- 5-Day Water Level Forecast (interactive spline charts)
- Precipitation Efficiency Comparison

### ⚡ Risk Predictor
- **6 input sliders**: Rainfall, Water Level, Snow Melt, Upstream Inflow, Duration, Hydrodynamic Index
- Real-time flood probability gauge (0–99%)
- Model diagnostics: RMSE, R², Accuracy
- Sub-model contribution breakdown (Snow Melt, Rainfall-Runoff, Flow Routing, Hydrodynamics)

### 🔗 FL Simulator
- Live **FedAvg** animation across 18 TensorFlow client models
- Configurable: training rounds, active clients, local epochs
- Round-by-round loss chart + accuracy ring gauge
- Network transfer tracker (weights-only, privacy-preserving)

---

## 🔒 Privacy-Preserving Design

```
Station 1 ──┐                          ┌── Station 1
Station 2 ──┤  weights only            ├── Station 2
    ...     ├──────────────►  Server  ──┤      ...
Station N ──┘  (no raw data)            └── Station N
                 FedAvg
```

Raw sensor data **never leaves** the local station. Only model weights are exchanged with the aggregation server.

---

## 📁 Project Structure

```
FFM/
├── app.py              # Streamlit dashboard (3 pages + CSS design system)
├── models.py           # TensorFlow FFNN + FedAvg federated learning
├── stations.py         # 18 Indian monitoring stations with simulated data
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

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
| `numpy` | Numerical computations |

---

## 🏞️ Monitored Rivers & Stations

| River | Stations |
|-------|----------|
| **Ganga** | Farakka Barrage, Haridwar, Varanasi, Patna, Prayagraj |
| **Brahmaputra** | Guwahati, Dibrugarh, Tezpur, Dhubri |
| **Yamuna** | Delhi (Old Rly Bridge), Mathura, Agra, Etawah |
| **Godavari** | Nashik, Rajahmundry, Polavaram |
| **Mahanadi** | Hirakud Dam, Naraj |

---

## 🧠 Model Details

### FFNN Architecture
```
Input (6 features)
    ↓
Dense(64, ReLU)
    ↓
Dense(32, ReLU)
    ↓
Dense(16, ReLU)
    ↓
Dense(1, Sigmoid) → Flood Probability
```

### Federated Learning (FedAvg)
1. Server broadcasts global model weights to all clients
2. Each client trains locally on its own sensor data
3. Clients send only updated weights back to server
4. Server averages all client weights (FedAvg)
5. Repeat for N rounds

---

## 📄 License

This project is licensed under the MIT License.