# 🚚 Supply Chain Intelligence Hub

**Predicting Late Delivery Risk in Supply Chain Management**  
Dataset: DataCo Smart Supply Chain | Models: Logistic Regression · Random Forest · XGBoost

---

## 📁 Files Included

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (run this) |
| `Supply_Chain_Complete.ipynb` | Full notebook with RF & XGBoost outputs |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run the Dashboard

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

The dashboard opens in your browser at `http://localhost:8501`

---

## 📊 Dashboard Tabs

| Tab | What's Inside |
|-----|--------------|
| 🏠 **Overview** | KPI cards, project intro, dataset preview |
| 🔍 **EDA** | Sales by delivery status, market, category, payment type, shipping mode, correlation heatmap |
| 🤖 **ML Models** | RF & XGBoost classification reports, confusion matrices, feature importances, model comparison |
| 🏭 **Suppliers** | Reliability rankings, on-time rates, risk labels per supplier/department |
| 🔮 **Predict** | Single shipment prediction form + batch CSV/Excel upload |

---

## 📤 Upload Your Own Company Data

- In the **sidebar**, select **"Upload company data"**
- Upload a `.csv` or `.xlsx` file only
- The file **must contain** a `Late_delivery_risk` column (0 = on-time, 1 = late)
- All models retrain automatically on your data

---

## 📓 Notebook Instructions

Open `Supply_Chain_Complete.ipynb` in Jupyter or Google Colab.

To load the dataset in Colab:
```python
import kagglehub
path = kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
data = pd.read_csv(f"{path}/DataCoSupplyChainDataset.csv", encoding="latin1")
```

Or place `DataCoSupplyChainDataset.csv` in the same folder and run locally.

---

## 🤖 Model Performance (on DataCo dataset)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 97.42% | 0.9612 | 0.9920 | 0.9764 |
| **Random Forest** | **97.44%** | **0.9624** | **0.9993** | **0.9805** |
| **XGBoost** | **97.44%** | **0.9637** | **0.9990** | **0.9810** |

---

## 🔑 Key Insights

1. `Days for shipping (real)` is the #1 predictor of late delivery risk
2. `Days for shipment (scheduled)` provides the model's understanding of planned timelines
3. `Order_to_Shipment_Time` (fulfillment speed) ranks 3rd — warehouse efficiency matters
4. **First Class** shipping has the highest late delivery rate despite its premium name
5. All three models achieve ~97.4% accuracy — ensemble methods (RF, XGBoost) edge out logistic regression
