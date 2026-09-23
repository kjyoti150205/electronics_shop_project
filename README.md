# 🛒 Electronic Shop Sales Prediction

A full-stack machine learning web application that predicts **order-level revenue**
for an electronics retail store based on product, ordering channel, customer state,
day of week, and month.

| Layer    | Technology                             |
|----------|----------------------------------------|
| ML       | scikit-learn, numpy                    |
| Backend  | Flask REST API (port 5000)             |
| Frontend | Streamlit, Plotly (port 8501)          |
| Data     | pandas, matplotlib, seaborn            |
| Report   | python-docx                            |

---

## Project Structure

```
electronics_shop_project/
├── data/
│   └── Electronics_Shop_Dataset.csv     # 11,000 transactions (Jan–Nov 2024)
├── model/
│   ├── model.pkl                        # Trained LinearRegression artefact
│   ├── scaler.pkl                       # StandardScaler (fitted on training data)
│   ├── le_product.pkl                   # LabelEncoder for product names
│   ├── le_channel.pkl                   # LabelEncoder for ordering channels
│   ├── le_state.pkl                     # LabelEncoder for US states
│   ├── metrics.json                     # Saved evaluation metrics
│   ├── model_info.json                  # Coefficients + class lists
│   └── diagnostics.png                  # Actual vs Predicted + Residual plots
├── backend/
│   └── app.py                           # Flask REST API
├── frontend/
│   └── ui.py                            # Streamlit UI (3 pages)
├── report_images/                        # Charts embedded in the Word report
├── train_model.py                        # Full ML training pipeline
├── generate_report.py                    # Generates Electronics_Shop_Report.docx
├── requirements.txt                      # All Python dependencies
└── README.md                             # This file
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
Saves `model/model.pkl`, `model/scaler.pkl`, all encoders, `metrics.json`,
`model_info.json`, `diagnostics.png`, and all EDA charts in `report_images/`.

### 3. Start the Flask backend (Terminal 1)
```bash
python backend/app.py
```
API runs at **http://localhost:5000**

### 4. Launch the Streamlit frontend (Terminal 2)
```bash
streamlit run frontend/ui.py
```
UI opens at **http://localhost:8501**

### 5. Generate the Word report
```bash
python generate_report.py
```
Produces **Electronics_Shop_Report.docx** in the project root.

---

## API Endpoints

| Method | Endpoint       | Description                                        |
|--------|----------------|----------------------------------------------------|
| GET    | `/health`      | Service health check                               |
| POST   | `/predict`     | Predict order revenue                              |
| GET    | `/dataset`     | KPIs, product/channel breakdown, monthly trend     |
| GET    | `/model_info`  | Model coefficients + evaluation metrics            |

### POST `/predict` — example

**Request:**
```json
{
  "product_name":        "Laptop",
  "channel_of_ordering": "Website",
  "state":               "California",
  "day_of_week":         0,
  "month":               6
}
```

**Response:**
```json
{
  "predicted_revenue":              "$1,842.50",
  "historical_avg_product_channel": "$1,750.32",
  "inputs": {
    "product_name":        "Laptop",
    "channel_of_ordering": "Website",
    "state":               "California",
    "day_name":            "Monday",
    "month":               6
  }
}
```

---

## Dataset

**Source:** [Kaggle — shafiirajabu/electronics-shop-dataset](https://www.kaggle.com/datasets/shafiirajabu/electronics-shop-dataset)

- **11,000** transactions
- **24 product categories** — Laptop, Smartphone, Drone, Smart TV, Tablet, …
- **5 ordering channels** — Website, Mobile App, Phone, Social Media, Physical
- **50 US States** — Jan–Nov 2024
- **Target:** `revenue = product_price × number_of_ordered_items`

---

## Model Performance

| Metric | Typical Value |
|--------|---------------|
| R²     | ~0.70         |
| MAE    | ~$150         |
| RMSE   | ~$200         |

---

## Frontend Pages

| Page              | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| Predict Revenue   | Input form → POST /predict → gauge chart + historical comparison  |
| Dataset Explorer  | KPI cards, top-products bar chart, channel pie, monthly trend     |
| Model Insights    | R²/MAE/RMSE cards, coefficients chart, diagnostics plot           |

---

*Electronic Shop Sales Prediction Project — IBM SkillsBuild*
