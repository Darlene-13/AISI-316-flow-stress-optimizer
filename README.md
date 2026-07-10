# AISI 316 Flow Stress Optimizer

An end-to-end machine learning project that uses an Artificial Neural Network (ANN) to predict and optimize the flow stress of AISI 316 stainless steel under hot deformation conditions.

## Overview

During hot deformation, the flow stress of stainless steel is influenced by three key parameters: temperature, strain rate, and strain. Classical approaches model this relationship using Arrhenius-type equations. This project replaces that manual fitting process with a trained multilayer perceptron (MLP) that learns the relationship directly from experimental data.

## Key Features

- Exploratory data analysis confirming physically consistent relationships across all input features
- Cubic spline interpolation along temperature and strain rate axes to expand the experimental dataset
- Five models trained and benchmarked: Linear Regression, SVR, Random Forest, XGBoost, and MLP
## Inputs & Output

| Input | Description |
|---|---|
| Temperature (T) | Deformation temperature in Kelvin |
| Strain Rate (ε̇) | Rate of deformation (s⁻¹) |
| Strain (ε) | Accumulated deformation |

| Output | Description |
|---|---|
| σ/σmax | Normalized flow stress |

## Model Comparison

| Model             | R²     | RMSE   | MAE    | MAPE   | MaxAE  |
|-------------------|--------|--------|--------|--------|--------|
| Linear Regression | 0.9128 | 0.0555 | 0.0437 | 0.0772 | 0.1721 |
| SVR               | 0.9872 | 0.0213 | 0.0159 | 0.0286 | 0.0691 |
| MLP (ANN)         | 0.9886 | 0.0201 | 0.0160 | 0.0249 | 0.0491 |
| Random Forest     | 0.9928 | 0.0159 | 0.0090 | 0.0167 | 0.0667 |
| XGBoost           | 0.9944 | 0.0141 | 0.0094 | 0.0175 | 0.0679 |

While XGBoost achieves the highest average performance, the MLP records the lowest worst-case prediction error across all five models which is critical in engineering applications where a single bad prediction carries real consequences. The ANN is also the only model that supports gradient-based optimization due to its smooth, differentiable prediction surface.

## Optimal Process Parameters

| Parameter   | Optimal Value |
|-------------|---------------|
| Temperature | 1051.4 °C     |
| Strain Rate | 1.2214 s⁻¹    |
| Strain      | 0.40          |
| Min σ/σmax  | 0.5341        |

These represent the most energy-efficient hot working conditions for AISI 304 stainless steel within the experimental operating range.

## Tech Stack

- **Python**: core language
- **TensorFlow / Keras**: ANN model
- **Scikit-learn**: preprocessing & evaluation
- **SciPy**: data interpolation & optimization
- **Pandas / NumPy**: data handling

## Project Structure

```
aisi304-flow-stress-optimizer/
│
├── data/
│   ├── raw/                            # Original experimental dataset
│   └── processed/                      # Interpolated & scaled splits
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_interpolation.ipynb          # Cubic spline interpolation
│   ├── 03_preprocessing.ipynb          # Scaling & train/val/test split
│   ├── 04_model_training.ipynb         # Baseline models + MLP training
│   └── 05_optimization.ipynb          # Surrogate-based optimization
│
├── src/
│   ├── data/                           # Loader & interpolator modules
│   ├── features/                       # Preprocessing pipeline
│   ├── model/                          # Architecture, trainer, optimizer
│   └── utils/                          # Metrics helpers
│
├── artifacts/
│   ├── scaler.pkl                      # Fitted MinMaxScaler
│   └── model.h5                        # Trained MLP
│
├── api/                                # FastAPI app (in progress)
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

## Project Status

| Stage                  | Status      |
|------------------------|-------------|
| Exploratory Analysis   | Complete    |
| Interpolation          | Complete    |
| Preprocessing          | Complete    |
| Model Training         | Complete    |
| Optimization           | Complete    |


## Material

AISI 316 Austenitic Stainless Steel

---

### Author
Darlene Wendy Nasimiyu