---
title: Melbourne Housing Price Predictor
emoji: 🏠
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Estimate Melbourne property prices with a trained ML pipeline
---

# Melbourne Housing Price Predictor

Estimate the market value of a Melbourne property from its suburb, property type,
bedrooms, distance from the CBD and location details.

This Space runs the same Streamlit app hosted on Streamlit Community Cloud, with
the trained scikit-learn pipeline committed as a pre-built artifact so the app is
ready as soon as the Space starts.

## Model

- **Algorithm:** Gradient Boosting (`HistGradientBoostingRegressor`)
- **Pipeline:** median/mode imputation + one-hot encoding + scaling bundled with the estimator
- **Features:** suburb, property type, region, postcode, bedrooms, distance from CBD, suburb property count
- **Target:** sale price (AUD)
- **Training data:** 25,154 cleaned Melbourne property sales (80/20 split)

| Metric | Test score |
| --- | ---: |
| R² | 0.707 |
| MAE | $202,474 |
| RMSE | $357,728 |
| MAPE | 18.7% |

## Links

- **Streamlit Cloud app:** <https://melbourne-housing-price-predictor-mwi93xzkp86ndxegowp5cq.streamlit.app>
- **Source code:** <https://github.com/ruksana-ux/melbourne-housing-price-predictor>

## Notes

Predictions are indicative estimates based on historic sales and do not account for
renovations, views, land/building size, year built or current market conditions.
They are not a substitute for a professional valuation.
