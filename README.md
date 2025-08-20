# Walmart Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DVC](https://img.shields.io/badge/MLOps-DVC-success)
![GitHub Actions](https://img.shields.io/badge/CI-CD%20Pipeline-blueviolet)

## 📌 Project Overview

This project aims to **forecast Walmart sales** using machine learning techniques. It follows an end-to-end **MLOps pipeline** integrating data ingestion, preprocessing, model training, evaluation, and versioning using **DVC (Data Version Control)**. The goal is to build a scalable, reproducible, and maintainable workflow for time series forecasting.

---

## 📂 Repository Structure

```
walmart_sale_forecast/
│── data/                # Raw and processed data (DVC-tracked)
│── logs/                # Application logs
│── notebooks/           # Jupyter notebooks for EDA and experiments
│   ├── raw_data.ipynb
│   └── model_creation.ipynb
│── pipeline/            # Orchestration scripts
│── src/                 # Core source code
│   ├── components/      # Data ingestion & preprocessing
│   ├── Model_creation/  # Model training & evaluation
│   ├── logger/          # Logging utilities
│   └── utils/           # Helper functions
│── requirements.txt     # Python dependencies
│── params.yaml          # Configurations and hyperparameters
│── dvc.yaml             # DVC pipeline definition
│── dvc.lock             # DVC lock file
│── .gitignore           # Git ignore rules
```

---

## 🚀 Features

* Automated **data ingestion & preprocessing**
* **Feature engineering** for time series forecasting
* **Model training & evaluation** (baseline + advanced models)
* **DVC integration** for data and model versioning
* **Logging system** for tracking experiments
* Modular code structure for **scalability & reusability**

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/walmart_sale_forecast.git
   cd walmart_sale_forecast
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install **DVC** (if not already):

   ```bash
   pip install dvc
   ```

---

## 📊 Usage

### 1. Run notebooks for exploration

```bash
jupyter notebook notebooks/raw_data.ipynb
```

### 2. Run pipeline stages

Use DVC to execute pipeline:

```bash
dvc repro
```

### 3. Train model manually

```bash
python src/Model_creation/model_creation.py
```

---

## ⚙️ Configuration

Modify hyperparameters and pipeline configs in `params.yaml`. Example:

```yaml
train:
  test_size: 0.2
  random_state: 42
model:
  type: xgboost
  n_estimators: 100
  learning_rate: 0.05
```

---

## 📈 Results

* Baseline model: Linear Regression
* Advanced models: XGBoost, Random Forest
* Metrics tracked: RMSE, MAE, R²

> Detailed experiment results are documented in the `notebooks/model_creation.ipynb`.

---

## 🔮 Future Work

* Hyperparameter optimization with Optuna
* Deploy model as a REST API using FastAPI/Flask
* Integration with MLflow for experiment tracking
* Automate CI/CD pipeline with GitHub Actions

---

## 🤝 Contribution

Contributions are welcome! To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.
