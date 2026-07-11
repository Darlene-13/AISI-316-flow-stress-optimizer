# AISI 316 Flow Stress Optimizer
## Objective
 
Determine the hot deformation processing parameters (temperature, strain
rate, strain) that minimize normalized flow stress (σ/σmax) for AISI 304
stainless steel, using trained ML surrogate models in place of physical
experimentation.

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
└── README.md
```

## Surrogate Model Comparison
 
Five regression models were trained and evaluated on the flow stress
dataset before any optimization was performed:
 
| Model              | R²     | RMSE   | MAE    | MAPE   | MaxAE  |
|---------------------|--------|--------|--------|--------|--------|
| Linear Regression    | 0.9128 | 0.0555 | 0.0437 | 0.0772 | 0.1721 |
| SVR                  | 0.9872 | 0.0213 | 0.0159 | 0.0286 | 0.0691 |
| MLP (ANN)            | 0.9886 | 0.0201 | 0.0160 | 0.0249 | 0.0491 |
| Random Forest        | 0.9928 | 0.0159 | 0.0090 | 0.0167 | 0.0667 |
| **XGBoost**          | **0.9944** | **0.0141** | 0.0094 | 0.0175 | 0.0679 |
 
XGBoost was the most accurate predictive model overall. However, model
accuracy alone does not determine which model can be used for
optimization — that depends on the optimization algorithm's requirements.
 
## Optimization Methodology
 
Two tracks were used, deliberately paired by mathematical compatibility
rather than by accuracy alone:
 
**Track 1: Gradient-based (L-BFGS-B) with the ANN surrogate.**
L-BFGS-B requires a differentiable objective function. Tree-based models
(XGBoost, Random Forest) produce discontinuous, step-like prediction
surfaces with no usable gradient, so they are mathematically incompatible
with gradient-based optimization. The ANN, being built entirely from
differentiable operations, was the only model in the comparison that
satisfies this requirement.
 
**Track 2: Derivative-free algorithms with the XGBoost surrogate.**
Since XGBoost was the most accurate model but not differentiable, four
derivative-free (black-box) optimization algorithms were used instead,
each representing a different search strategy:
 
- **Differential Evolution**: population-based, evolves candidate
  solutions via mutation and crossover (genetic algorithm family).
- **Particle Swarm Optimization**: a swarm of candidate solutions moves
  through the search space, pulled toward the best positions found.
- **Simulated Annealing**: a single random walk that gradually "cools,"
  accepting worse moves early on to escape local minima.
- **Bayesian Optimization**: builds a probabilistic model of the
  objective and chooses new points to evaluate based on expected
  improvement; most valuable when each evaluation is expensive.
All searches were constrained to the physical bounds of the experimental
dataset:
 
| Parameter       | Lower Bound   | Upper Bound   |
|------------------|---------------|---------------|
| Temperature (1/T)| 0.000755 K⁻¹  | 0.000817 K⁻¹  |
| ln(Strain Rate)  | -2.3026       | 2.7081        |
| Strain           | 0.1           | 0.7           |
 
## Results
 
| Algorithm | Surrogate | Temperature (°C) | Strain Rate (s⁻¹) | Strain | σ/σmax | Time (s) |
|---|---|---|---|---|---|---|
| L-BFGS-B | ANN | 1051.4 | 1.2214 | 0.40 | 0.5333 | — |
| Differential Evolution | XGBoost | 1039.2 | 0.1237 | 0.10 | **0.1755** | 1.30 |
| Particle Swarm | XGBoost | 1031.7 | 0.4391 | 0.11 | **0.1755** | 4.71 |
| Simulated Annealing | XGBoost | 1000.3 | 0.2113 | 0.11 | **0.1755** | 1.34 |
| Bayesian Optimization | XGBoost | 1006.9 | 0.4818 | 0.12 | **0.1755** | 23.59 |
 
## Key Findings
 
**1. Cross-algorithm convergence validates the global minimum.**
All four derivative-free methods, despite using unrelated search
strategies (population-based, swarm-based, stochastic annealing, and
probabilistic modeling), independently converged on the same objective
value, σ/σmax = 0.1755. Agreement across four unrelated algorithms is
strong evidence this is the true global minimum of the XGBoost surrogate
over the given bounds, not an artifact of any single method.
 
**2. The gradient-based result is comparatively worse, and why.**
L-BFGS-B converged to σ/σmax = 0.5333, notably higher (worse) than the
XGBoost-based result. Its optimal strain rate and strain values are
nearly identical to the initial guess (`x0`), indicating the search barely
left the neighborhood of its starting point. This is expected behavior
for local gradient-based optimizers, they are sensitive to initialization
and can converge on a local rather than global minimum. It is not a
reflection of the ANN being a poor model, but of pairing a local search
method with a single starting point.
 
**3. Runtime efficiency favors the simpler global searches here.**
Bayesian Optimization took 23.59s to reach the same result Differential
Evolution found in 1.30s. Bayesian Optimization's strength is minimizing
the *number* of objective evaluations, which matters when each evaluation
is expensive (e.g., physical experiments or slow simulations). Since the
XGBoost surrogate evaluates near-instantly, that advantage does not
translate into a runtime benefit here; it only adds overhead.
 
**4. Same output, different inputs: a plateau in the XGBoost surface.**
The four XGBoost-based methods agree on σ/σmax to four decimal places,
yet land on visibly different input combinations (temperature spans
1000–1039°C, strain rate spans 0.12–0.48 s⁻¹). This is consistent with
tree-based ensembles producing flat, plateau-like prediction regions
rather than a single smooth minimum: once strain is pushed near its lower
bound, a range of temperature/strain-rate combinations fall into the same
region of the ensemble's decision structure and predict the same output.
This is a property of the model's surface, not noise or search error.
 
## Recommended Optimal Parameters
 
Based on the convergence agreement across all four global search methods,
the recommended optimal hot deformation parameters are:
 
| Parameter | Value |
|---|---|
| Temperature | ≈ 1000–1039 °C |
| Strain Rate | ≈ 0.12–0.48 s⁻¹ |
| Strain | ≈ 0.10 |
| Predicted σ/σmax | 0.1755 |
 
**Differential Evolution's result (1039.2 °C, 0.1237 s⁻¹, strain 0.10) is
put forward as the single reported optimum**, since it is the fastest of
the four methods to reach the validated minimum and belongs to the
well-established genetic algorithm family.
 
## Conclusion
 
Minimizing flow stress is the primary engineering objective in hot
deformation processing: lower flow stress means less resistance to
deformation, reduced energy and force requirements, less tool wear, and
higher overall process efficiency. This study shows that the choice of
optimization algorithm is not independent of the choice of surrogate
model, gradient-based optimization constrains the surrogate to
differentiable models (ANN) regardless of their predictive accuracy,
while derivative-free algorithms free the optimization to use the most
accurate available model (XGBoost). Four independent derivative-free
algorithms converging on the same minimum gives high confidence that
σ/σmax ≈ 0.1755, achieved near a strain of 0.10 and a temperature between
1000 - 1039°C, represents the true optimal processing window for AISI 304
within the tested experimental bounds, a substantially lower, and more
trustworthy, result than the local optimum found by gradient-based search
on the less accurate ANN surrogate.


## Material

AISI 316 Austenitic Stainless Steel

---

### Author
Darlene Wendy Nasimiyu