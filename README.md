# Melbourne Housing Price Predictor

A machine learning web app that estimates the market value of a Melbourne property.
The app is built with **scikit-learn** and **Streamlit** and is ready to deploy on
**Streamlit Community Cloud** straight from GitHub.

## Demo

Enter a suburb, property type, number of bedrooms, distance from the CBD and a few
location details. The app returns an estimated price with a likely range based on the
model's average test error.

## Project structure

```
melbourne-housing-streamlit/
├── app.py                      # Streamlit application
├── train_model.py              # Trains, evaluates and saves the model
├── requirements.txt            # Pinned Python dependencies
├── runtime.txt                 # Python version for Streamlit Cloud
├── .streamlit/config.toml      # Theme and server settings
├── data/
│   └── Melbourne_Housing.csv   # Training data (25,154 cleaned rows)
└── model/
    ├── housing_model.joblib    # Deployment-ready scikit-learn pipeline
    └── model_metadata.json     # Metrics, dropdown options and suburb defaults
```

## Model

The pipeline bundles imputation, one-hot encoding, scaling and the estimator, so the
exact same transformations run during training and inference.

| Model | Test R² | Test MAE | Test RMSE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.626 | $248,902 | $403,722 | 26.6% |
| Random Forest | 0.696 | $204,845 | $364,398 | 18.7% |
| **Gradient Boosting (deployed)** | **0.707** | **$202,474** | **$357,728** | **18.7%** |

**Features:** suburb, property type, region, postcode, bedrooms, distance from the
CBD, and number of properties in the suburb.

**Target:** sale price in AUD.

Evaluation uses an 80/20 train-test split with a fixed random seed (42), after dropping
rows with missing required fields and duplicate records.

## Run locally

The project uses Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

uv pip install -r requirements.txt
streamlit run app.py
```

Or with pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Retrain the model

```bash
python train_model.py
```

This rebuilds `model/housing_model.joblib` and `model/model_metadata.json`, comparing
Linear Regression, Random Forest and Gradient Boosting. The best test R² is saved
automatically, so the app picks up the new artifact with no code changes.

## Deploy on GitHub + Streamlit Community Cloud

### 1. Push the project to GitHub

Create a new repository at <https://github.com/new> (for example
`melbourne-housing-price-predictor`). Do **not** initialise it with a README, then run:

```bash
git init
git add .
git commit -m "Build Melbourne housing price predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Make sure `.venv/` is not committed — it is already excluded by `.gitignore`.

### 2. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** (or **New app**) and choose **Deploy a public app from GitHub**.
3. Select your repository, the `main` branch and set the main file path to `app.py`.
4. In **Advanced settings**, choose **Python 3.12**.
5. Click **Deploy**. The first build installs `requirements.txt` and usually takes a
   few minutes.
6. Share the generated `https://<app-name>.streamlit.app` URL.

Every push to `main` triggers an automatic redeploy. If you change the dataset or
retrain the model, commit the updated `model/` files.

## Notes and limitations

- The dataset is a historic Melbourne sales snapshot; prices are not adjusted for
  inflation or current market conditions.
- Land size, building area, year built and property condition are not included, which
  limits accuracy for unusual or premium properties.
- Predictions are indicative only and are not a substitute for a professional
  valuation.
