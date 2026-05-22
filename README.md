# ⚡ LoadIQ — Smart AI Energy Forecasting Platform

<div align="center">

![LoadIQ Banner](https://img.shields.io/badge/LoadIQ-Smart%20AI%20Energy-00d4ff?style=for-the-badge&logo=lightning&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-loadiq--smart--ai.web.app-22c55e?style=for-the-badge&logo=firebase&logoColor=white)](https://loadiq-smart-ai.web.app/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Model-ff6b35?style=for-the-badge)

**Not just a dashboard. A thinking energy system.**

[🌐 Live Demo](https://loadiq-smart-ai.web.app/) · [📖 Backend Docs](#backend) · [🎨 Frontend Docs](#frontend) · [📊 Model Docs](#model-architecture)

</div>

---

## 📌 What is LoadIQ?

LoadIQ is an AI-powered electricity load forecasting platform that predicts real-time energy demand across four countries — **UK, USA, Germany, and India** — using machine learning models trained on six years of half-hourly transmission-level grid data.

Unlike conventional energy monitoring tools that pre-load everything and waste server resources, LoadIQ is engineered to be as efficient as the sustainable future it helps build.

> **Accurate forecasting → less over-generation → less carbon waste.**
> That is the principle LoadIQ is built on.

---

## 🏗️ Project Structure

```
LOADIQ/
│
├── Backend/                        # FastAPI prediction servers
│   ├── Model1.py                   # DE+LU load prediction API  (port 8001)
│   ├── Model2.py                   # GB load prediction API     (port 8002)
│   └── README.md                   # Backend setup & API docs
│
├── energy-analytics/               # React frontend (TypeScript)
│   ├── src/
│   │   ├── components/             # Dashboard, charts, chatbot UI
│   │   ├── services/
│   │   │   ├── api.js              # Unified API layer with fallback
│   │   │   ├── ChatbotService.js   # Gemini AI integration
│   │   │   └── AnalyticsService.js # Data processing & telemetry
│   │   └── pages/                  # Route-level page components
│   └── README.md                   # Frontend setup & deployment docs
│
├── DATA_preprocessing/
│   ├── Modelling/                  # Saved model artifacts (.pkl)
│   └── energy_model_upgraded.ipynb # Full model training notebook
│
├── DATA/
│   ├── time_series_60min_cleaned.csv    # DE+LU · 50,401 rows · 45 cols
│   └── energy_30min_processed.csv       # GB · 100,802 rows · 46 cols
│
└── README.md                       # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

### 1. Clone the repository

```bash
git clone https://github.com/your-username/loadiq.git
cd loadiq
```

### 2. Train the models (first time only)

Open and run all cells in:
```
DATA_preprocessing/energy_model_upgraded.ipynb
```

This saves 6 model artifacts to `DATA_preprocessing/Modelling/`:
```
lgb_load_model.pkl       xgb_load_model.pkl
ridge_load_model.pkl     X_scaler.pkl
feature_names.pkl        train_medians.pkl
```

### 3. Start the backend servers

```bash
# Terminal 1 — DE+LU model (port 8001)
cd Backend
pip install -r requirements.txt
python Model1.py

# Terminal 2 — GB model (port 8002)
python Model2.py
```

### 4. Start the frontend

```bash
cd energy-analytics
npm install
npm start
# Opens http://localhost:3000
```

---

## 🤖 Model Architecture

```
Raw Data (55 features)
        │
        ▼
┌─────────────────────────────────────┐
│        Feature Engineering          │
│  • Lag features: 1h, 24h, 168h ago  │
│  • Rolling mean/std: 24h, 168h       │
│  • Time: hour, day_of_week, month    │
└─────────────────────────────────────┘
        │
        ▼
┌──────────────┐    ┌──────────────┐
│  LightGBM    │    │   XGBoost    │
│  (leaf-wise) │    │  (level-wise)│
└──────┬───────┘    └──────┬───────┘
       │  60%              │  40%
       └────────┬──────────┘
                ▼
        ┌──────────────┐
        │   Ensemble   │    R² > 99%
        │   Output     │    MAPE < 1.5%
        └──────────────┘
                │
                ▼
        Predicted Load (MW)
```

### Training Data

| Dataset | Interval | Rows | Columns | Countries |
|---------|----------|------|---------|-----------|
| `time_series_60min_cleaned.csv` | 60 min | 50,401 | 45 | Germany, Luxembourg |
| `energy_30min_processed.csv` | 30 min | 100,802 | 46 | GB, Cyprus, Ireland |
| **Date range** | — | — | — | 2015 – 2020 |
| **Null values** | — | — | — | Zero |

---

## 🧠 Smart Engineering Decisions

| Decision | Reason |
|----------|--------|
| **Demand-based page loading** | Pages render only when navigated to — no pre-fetch, no wasted bandwidth |
| **Skeleton-first UI** | Interface feels instant even before API data arrives |
| **Median-fill inference** | Missing features filled with training medians, not zeros — preserves model integrity |
| **Chronological train/test split** | Never trains on future data — prevents data leakage |
| **TimeSeriesSplit CV** | 5-fold validation that always respects time order |
| **Ensemble output** | Blends LightGBM + XGBoost to smooth variance at peak transitions |
| **Heuristic fallback** | If ML model fails, weighted heuristic returns reasonable estimate — no crash |
| **Firebase error logging** | All backend exceptions logged to Firestore for monitoring |

---

## 🌍 Live Demo

**🔗 [https://loadiq-smart-ai.web.app/](https://loadiq-smart-ai.web.app/)**

Hosted on Firebase Hosting. The frontend is always live. For full prediction functionality, the backend servers must be running locally or deployed separately.

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Three.js, Recharts |
| Backend | FastAPI, Uvicorn, Python 3.10 |
| ML Models | LightGBM, XGBoost, Ridge Regression, scikit-learn |
| AI Chatbot | Google Gemini 2.0 Flash API |
| Data | ENTSO-E Transparency Platform |
| Hosting | Firebase Hosting |
| Error Logging | Firebase Firestore |

---

## 👨‍💻 Developer

**Parth Sanjay Mhatre**
Pillai College of Engineering, Mumbai University — Semester 6

---

## 📄 License

This project is developed for academic and demonstration purposes.
