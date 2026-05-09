# Car Price Prediction using Machine Learning

This project focuses on predicting the resale prices of second-hand cars using machine learning. We used a dataset with over 10,000 entries containing car features such as make, model, year, mileage, and more. By applying and comparing different regression models, we were able to generate insights to help inform pricing strategies and evaluate business profitability.

---

## Key Highlights

- Processed 10,000+ car listings using `Pandas` and `NumPy`
- Applied ML models including `XGBoost`, `LightGBM`, and `Random Forest`
- Achieved high accuracy using optimized regression techniques and feature engineering
- Evaluated model performance using metrics such as **RMSE** and **R²**
- Analyzed **feature importance** to determine key drivers of car price
- Explored relationships between year, mileage, brand, and pricing using `Seaborn` and `Matplotlib`

---

## Technologies and Tools

- Python
- Pandas, NumPy
- scikit-learn
- XGBoost, LightGBM
- Matplotlib, Seaborn
- Jupyter Notebook

## Motivation

Have been recently attempting to sell my car in the second hand market and have been getting increasilngly frustrated with the low prices my car has been receiving on websites online. Hence, I decided to explore how much my car was worth my trianing a ML model on a large dataset of second hand cars.

---

## Using the trained model outside the notebook

The notebook `FinalML.ipynb` is great for experimentation. For **reuse in apps or services**, install dependencies and train a persisted pipeline:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m car_pricing.train --data car_price_dataset.csv --out artifacts
```

Optional: **`pip install xgboost`** then **`--backend xgboost`** (same preprocessor as **`FinalML.ipynb`**). On Apple Silicon/macOS OpenMP errors are common unless **`brew install libomp`** is installed.

Default training backend is **`sklearn_hist`** (`HistGradientBoostingRegressor`): no native extra libraries, same preprocessing, still joblib-saveable end-to-end.

Artifacts:

- **`artifacts/pipeline.joblib`** — full sklearn `Pipeline` (preprocessing + boosted trees), ready to load in Python or any joblib-compatible runtime.
- **`artifacts/metadata.json`** — holdout RMSE/MAE/R² and feature lists for auditing.

**Batch scoring (CSV or JSON):**

```bash
python -m car_pricing.predict artifacts/pipeline.joblib --csv new_listings.csv
python -m car_pricing.predict artifacts/pipeline.joblib --json '{"Brand":"Toyota","Model":"Camry","Year":2015,"Engine_Size":2.5,"Fuel_Type":"Petrol","Transmission":"Automatic","Mileage":90000,"Doors":4,"Owner_Count":2}'
```

**HTTP API (integrate with web apps, mobile backends, or other services):**

```bash
uvicorn car_pricing.api:app --host 0.0.0.0 --port 8000
```

Example:

```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Brand":"Toyota","Model":"Camry","Year":2015,"Engine_Size":2.5,"Fuel_Type":"Petrol","Transmission":"Automatic","Mileage":90000,"Doors":4,"Owner_Count":2}'
```

`POST /predict/batch` accepts a JSON array of the same objects. Docs: `http://127.0.0.1:8000/docs` when the server is running.

Optional env vars:

- **`CAR_PRICE_MODEL_PATH`** — path to `pipeline.joblib` (default `artifacts/pipeline.joblib`).
- **`CAR_PRICE_ARTIFACT_DIR`** — defaults to `artifacts` (metadata loading for `/health`).

For notebook work, continue using `FinalML.ipynb`; wire production scoring to **`car_pricing`** so preprocessing stays identical across training and inference.

---

## Tests, pre-commit, and CI

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/
pre-commit install
pre-commit run --all-files   # trains on tests/fixtures/tiny_car_prices.csv, then pytest
```

On push/PR to **`main`** or **`master`**, [`.github/workflows/ci.yml`](.github/workflows/ci.yml) installs dependencies, runs a **tiny CSV train**, **`pytest tests/`**, and **`pre-commit run --all-files`** so CI matches local hooks.

