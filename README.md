# Car Price Prediction: Geely Auto (US Market)

A multiple linear regression model that predicts car prices from vehicle specifications, built to help Geely Auto decide how to price its cars when entering the US market.

## Problem
Geely wants to enter the US car market and needs to know **which features drive car prices** and **how to price accordingly**. The goal is a model that predicts price from car features and shows which variables matter most.

## Results (held-out test set, 41 cars)

| Metric | Value |
|---|---|
| R² | **0.946** |
| MAE | **$1,454** |
| RMSE | $2,069 |
| Train R² | 0.964 (small train-test gap, so little overfitting) |
| 5-fold CV R² (log price) | **0.908 ± 0.019** |

**Improvement from log-transforming price:** the baseline model on raw price scored test R² 0.902 and MAE $1,911. Log-transforming the right-skewed target (skew 1.78) raised test R² to 0.946 and cut MAE by about 24%. Metrics above are reported back on the original price scale.

## What Drives Price
- **Brand matters most.** Porsche and BMW carry the largest price premiums. Mitsubishi and Plymouth carry the largest discounts (all relative to the reference brand).
- **Body style:** wagons and hatchbacks are priced lower than the reference body type.
- **Curb weight** is the strongest numeric driver. Horsepower, engine size and car width are also strongly correlated with price.
- Fuel efficiency (city and highway mpg) is negatively correlated with price.

## Approach
1. **Cleaning:** extracted the brand from `CarName` and fixed misspellings (e.g. `maxda` to `mazda`, `toyouta` to `toyota`, `vw` to `volkswagen`). No missing values or duplicates in the 205 rows.
2. **EDA:** distributions, skewness, scatter plots against price and a correlation heatmap.
3. **Preprocessing:** 80/20 split, `StandardScaler` for numeric features and one-hot encoding for categorical features, wrapped in a scikit-learn pipeline.
4. **Multicollinearity:** used VIF to find redundancy, such as `fuelsystem` duplicating `fueltype`. Dropped `fuelsystem`, and tested removing `enginelocation`, `enginetype`, `cylindernumber` and the strongly correlated numeric features. Removing too many variables lowered test R², so the final model keeps the strongest predictors and accepts some multicollinearity.
5. **Final model:** linear regression on log-price with 5-fold cross-validation, plus residual analysis and a review of the worst predictions.

## Limitations
- Only **205 cars**, so test metrics come from 41 cars and can vary with the split. The cross-validated R² is the steadier estimate.
- The largest single error is about 39% on one car. Residuals are checked in the notebook.
- Some correlated predictors remain, so individual coefficients should be read with care. The model is better for prediction than for exact effect sizes.

## Tech Stack
Python · pandas · NumPy · scikit-learn · statsmodels · matplotlib · seaborn

## Run It
```bash
git clone https://github.com/tanmayanju16616-web/geely-car-price-prediction.git
cd geely-car-price-prediction
pip install -r requirements.txt
jupyter notebook Geely_Car_Price_Prediction.ipynb
```
The dataset (`CarPrice_Assignment.csv`) is not included. Place it in the same folder as the notebook.

## Author
**Anju Kanwar**
