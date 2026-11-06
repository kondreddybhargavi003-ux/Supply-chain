# 📦 StockSense – Supply Chain Intelligence & Inventory Optimization

**An End-to-End Data Science and Machine Learning Project**

StockSense is a data-driven supply chain intelligence and inventory optimization project designed to help retail businesses make smarter inventory decisions using historical sales, inventory, and procurement data.

The project focuses on understanding product demand, identifying potential stockout risks, analyzing inventory patterns, and supporting better procurement decisions through Data Science and Machine Learning.

---

## 🎯 Project Objectives

- **Demand Forecasting:** Predict future product demand using historical sales data.
- **Stockout Prediction:** Identify products that may run out of stock and help businesses take preventive action.
- **Inventory Optimization:** Support inventory planning using demand forecasts and available stock information.
- **Procurement Analysis:** Analyze suppliers, purchase orders, goods receipts, and supplier invoices.
- **Data-Driven Decisions:** Transform raw business data into meaningful insights and predictions.

---

## 📊 Datasets

The project uses retail and procurement datasets collected from external data sources.

### 1. Retail Datasets

| Dataset | Records | Columns |
|---|---:|---:|
| Retail Inventory | 284,755 | 18 |
| Retail Sales | 125,751 | 17 |

**Inventory dataset includes:**
- Product and supplier information
- Store and sales channel information
- Available stock quantity
- Stock selling amount and cost
- Inventory start and end dates

**Sales dataset includes:**
- Transaction dates
- Product and store information
- Quantity sold
- Sales amount and cost of goods sold
- Sales type and return indicators

### 2. Procurement Datasets

The procurement dataset consists of eight related CSV files.

| Dataset | Records |
|---|---:|
| Buyers | 5 |
| Goods Receipts | 89 |
| Items | 20 |
| PO Lines | 91 |
| Purchase Orders | 36 |
| Requisitions | 150 |
| Suppliers | 14 |
| Supplier Invoices | 87 |

These datasets provide information about purchasing activities, suppliers, procurement requests, goods receipts, and invoice processing.

**Note:** Retail and procurement datasets are currently treated as separate data sources because a verified mapping between their product identifiers has not yet been established.

---

## 🏗️ Project Workflow

The project follows an end-to-end Data Science and Machine Learning workflow.

1. **Dataset Collection**
   - Collect retail sales, inventory, and procurement datasets.
   - Organize raw files into structured directories.

2. **Data Understanding**
   - Explore dataset structure and dimensions.
   - Understand columns, data types, and business meaning.

3. **Data Quality Analysis**
   - Identify missing values.
   - Check duplicate records.
   - Validate dates and data relationships.
   - Investigate inconsistent records.

4. **Data Preprocessing**
   - Handle missing and invalid values.
   - Convert appropriate data types.
   - Clean and prepare datasets for analysis.

5. **Exploratory Data Analysis (EDA)**
   - Analyze sales trends.
   - Study product and store performance.
   - Explore inventory patterns.
   - Analyze procurement activities.

6. **Feature Engineering**
   - Create time-based features.
   - Generate lag and rolling statistical features.
   - Develop inventory-related indicators.

7. **Machine Learning**
   - Build demand forecasting models.
   - Develop stockout prediction models.
   - Compare candidate algorithms.

8. **Model Evaluation**
   - Evaluate forecasting performance.
   - Assess classification performance.
   - Compare models and select suitable approaches.

9. **Application Development**
   - Develop an interactive Streamlit application.
   - Display demand forecasts, inventory insights, and stockout risk predictions.

---

## 🤖 Planned Machine Learning Models

The following are candidate algorithms to be evaluated during experimentation.

### Demand Forecasting
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Suitable time-series forecasting methods

### Stockout Prediction
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Final models will be selected based on experimental results and appropriate validation.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Advanced ML | XGBoost |
| Application Development | Streamlit |
| Development Environment | Antigravity IDE |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
StockSense/
│
├── app/
│   └── Streamlit application
│
├── data/
│   ├── raw/
│   │   ├── Inventory/
│   │   ├── Sales/
│   │   └── Procurement/
│   │
│   └── processed/
│       ├── sales_clean.csv
│       └── inventory_intervals_clean.csv
│
├── models/
│   └── Trained model files
│
├── notebooks/
│   └── Data exploration and analysis
│
├── reports/
│   └── Analysis and preprocessing reports
│
├── scripts/
│   └── Data processing scripts
│
├── .gitignore
├── requirements.txt
└── README.md
```

*The structure will evolve as the project progresses.*

---

## 📈 Expected Outcomes

The final StockSense system aims to provide:

- Product-level demand forecasts.
- Early identification of potential stockout risks.
- Inventory planning insights.
- Supplier and procurement analysis.
- Interactive visualizations and business insights.
- A machine learning-based decision-support application.

---

## 🚧 Project Status

**Currently under development.**

| Phase | Status |
|---|---|
| Dataset Collection | Completed |
| Initial Data Understanding | Completed |
| Data Quality Analysis | Completed |
| Relationship and Date Checks | Completed |
| Data Preprocessing | In Progress |
| Exploratory Data Analysis | Upcoming |
| Feature Engineering | Upcoming |
| Machine Learning | Upcoming |
| Model Evaluation | Upcoming |
| Streamlit Application | Planned |

---

## 🚀 Future Enhancements

- Advanced demand forecasting.
- Improved stockout risk estimation.
- Inventory reorder recommendations.
- Supplier performance evaluation.
- Interactive inventory monitoring.
- Model explainability and prediction insights.

---

## 👩‍💻 Author

**Bhargavi**

GitHub: [@kondreddybhargavi003-ux](https://github.com/kondreddybhargavi003-ux)

---

⭐ If you find this project interesting, feel free to explore the repository and follow its development.

**StockSense – Turning Supply Chain Data into Smarter Inventory Decisions.**