# Decarbonizing the Built Environment: Identifying High-Impact Opportunities within the Building Data Genome Project 2 (BDG2)

---

## The Context: Staying Within the 1.5°C Carbon Budget

The UN Environment Programme (UNEP) Emissions Gap Report 2025 states that the remaining carbon budget for a 66% likelihood of limiting warming to 1.5°C is only approximately 80 GtCO₂, a budget expected to be exhausted before 2030 at current emission rates of roughly 40 GtCO₂ annually (Forster et al., 2025; UN Environment Programme, 2025). As the built environment is a major contributor to global energy consumption and associated CO₂ emissions, it offers substantial potential for immediate and scalable reductions. This project directly addresses this potential by employing machine learning (ML) regression techniques to pinpoint specific inefficiencies within real-world building data, providing actionable insights for targeted carbon reduction strategies.

<ins>References:</ins>

Forster, P. M., Smith, C., Walsh, T., Lamb, W. F., Lamboll, R., Cassou, C., et al. (2025). Indicators of global climate change 2024: annual update of key indicators of the state of the climate system and human influence

UN Environment Programme. (2025). Emissions gap report 2025: Off target — Continued collective inaction puts global temperature goal at risk. Nairobi.

---

## Project Goal

The primary goal of this project is to analyze the energy consumption data of non-residential buildings provided by the BDG2 dataset to:

1. Establish a predictive baseline for building Energy Use Intensity (EUI)
2. Identify buildings that are significantly underperforming compared to similar buildings
3. Identify the drivers of energy inefficiency in buildings
4. Provide actionable recommendations for CO₂ reduction

---

## Key Findings

### Model Performance

| Model | Train R² | Test R² |
|-------|----------|---------|
| Linear Regression | 0.428 | 0.437 |
| **Tuned Random Forest** | **0.808** | **0.702** |
| Tuned LightGBM | 0.794 | 0.711 |

The tuned models explain **~70% of EUI variance**, with LightGBM slightly outperforming Random Forest. This represents a significant improvement over linear regression (44% R²).

> **Note on Model Scope:** While the model uses only building characteristics (type, region, meter, year built, climate) without explicit site identifiers, validation shows it cannot generalize to entirely new sites (group split R² < 0). The model is best suited for benchmarking buildings within an existing portfolio, not predicting EUI for new locations.

### Savings Potential Analysis

From 1,166 test buildings, **649 underperformers (55.7%)** were identified with a total savings potential of **4.95 million kWh/sqm**.

| Breakdown | Savings (kWh/sqm) | % of Total |
|-----------|-------------------|------------|
| **Chilledwater (cooling)** | 3,391,526 | **68.5%** |
| **Hotwater (heating)** | 1,371,099 | **27.7%** |
| Gas | 50,685 | 1.0% |
| Electricity | 37,709 | 0.8% |

### Key Insight

**Thermal systems (chilledwater + hotwater) account for 96% of all savings potential.** Direct electricity and gas inefficiency is minimal.

### Priority Targets

- **Region**: City of Westminster, England — 90.9% of all savings potential
- **Building Type**: Education — 42.1% of total savings
- **Thermal systems**: Chilledwater + hotwater combined = 96.2% of savings potential

---

## Dataset

**Building Data Genome Project 2 (BDG2)**

BDG2 is an open dataset made up of 3,053 energy meters from 1,636 buildings. The time range of the time-series data is two full years (2016-2017) with hourly measurements of electricity, heating and cooling water, steam, and irrigation meters.

The dataset consists of three primary components:

- **Building Metadata:** Building types, locations, characteristics (floor area, building age, facility type)
- **Meter Data:** Energy consumption readings for electricity, chilled water, steam, hot water, and gas
- **Weather Information:** Local weather data including temperature, humidity, and wind speed

**Dataset Source:** [BDG2 GitHub Repository](https://github.com/buds-lab/building-data-genome-project-2)

---

## Methodology

A cross-sectional regression approach was employed:

1. **Data Aggregation:** Hourly meter readings were summed to annual totals per building-meter-year combination, then converted to Energy Use Intensity (EUI = kWh/sqm) which is the total meter reading normalized by the building's square meter
2. **Target Variable:** `log_eui` — log-transformed EUI using `log1p(eui)` to handle skewed distribution
3. **Features Used:**
   - Categorical: `primaryspaceusage`, `region`, `meter`
   - Numerical: `yearbuilt`, `Avg_AirTemp_Annual`
3. **Missing Data Handling:**
   - `region`: Temperature-based imputation (before train/test split)
   - `yearbuilt`: Group-wise median imputation by site (after split to prevent leakage)
4. **Modeling:** Random Forest with hyperparameter tuning via RandomizedSearchCV (5-fold CV)
5. **Benchmark:** Linear Regression as baseline to quantify ensemble model improvement
6. **Validation:** LightGBM comparison to confirm robustness
7. **Efficiency Analysis:** Buildings with positive residuals (actual > predicted) flagged as underperformers

---

## Repository Structure


    .
    ├── jupyter_notebooks/
    │   ├── meta_data_preprocessing.ipynb      # Metadata cleaning
    │   ├── meter_data_cleaned_preprocessing.ipynb  # Cleaned meter aggregation
    │   ├── weather_preprocessing.ipynb        # Weather data processing
    │   ├── EDA                                # Visualizing important characteristics of the dataset
    │   └── modeling_final.ipynb               # Final modeling notebook ⭐
    ├── data/
    │   ├── df_analysis.csv                    # Dataset for the analysis
    ├── report/
    │   ├── final_report.md                    # Technical report & AI ethical considerations
    │   └── figures/                           # Exported visualizations
    │       ├── model_comparison.png           # Model performance comparison
    │       ├── predicted_vs_actual.png        # Prediction scatter plot
    │       ├── residuals_distribution.png     # Residuals histogram
    │       ├── feature_importance.png         # RF feature importances
    │       ├── shap_summary.png               # SHAP value summary
    │       ├── regional_bias.png              # Regional bias analysis
    │       ├── savings_by_meter.png           # Savings by meter type
    │       └── savings_by_region.png          # Savings by region
    └── README.md


---

## Key Technologies & Libraries

- **Python 3.x**
- **Pandas** — Data manipulation and aggregation
- **NumPy** — Numerical operations
- **Scikit-learn** — Modeling, preprocessing, cross-validation
- **LightGBM** — Gradient boosting for model validation
- **SHAP** — Model interpretability and feature importance
- **Matplotlib/Seaborn** — Visualization

---

## Getting Started

### Prerequisites

\`\`\`bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn
\`\`\`

### Running the Analysis

1. **Modeling:** Open and run `modeling_final.ipynb` — this is the main analysis notebook containing:
   - Model training and hyperparameter tuning
   - Feature importance analysis (SHAP values)
   - Cross-validation and error metrics (R², RMSE, MSE)
   - Regional bias analysis
   - Savings potential calculations by building type, region, and meter
   - Top 10 priority buildings for retrofit
   - Limitations and recommendations

2. **Final Report:** See `report/final_report.md` for the complete technical report including:
   - Executive summary for stakeholders
   - Technical pipeline documentation
   - AI ethical considerations and fairness analysis

---

## Recommendations

Based on the analysis, the following actions are recommended for maximum CO₂ reduction impact:

1. **Prioritize HVAC retrofits** — 96% of savings comes from thermal systems (chilledwater + hotwater)
2. **Focus on Westminster** — 90.9% of savings concentrated in one region
3. **Target Education buildings** — Largest share (42.1%) of total savings potential
4. **Optimize chiller plants first** — Chilledwater alone accounts for 68.5% of all savings
5. **Conduct energy audits on top 10 buildings** — Captures ~53% of total savings potential

---

## Model Validation & Robustness Checks

To ensure the model is valid for portfolio benchmarking, multiple splitting strategies were tested with Random Forest:

| Split Strategy | RF Test R² | Purpose |
|----------------|------------|---------|
| Random Split | 0.702 | Baseline performance |
| **Temporal Split (2016→2017)** | **0.750** | Predict same buildings across years |
| Group Split (new sites) | -0.08 | Test generalization to unseen sites |

**Key Findings:**
- ✅ **Temporal split validates portfolio benchmarking** — The model achieves 75% R² when trained on 2016 data and tested on 2017 data for the same buildings
- ✅ **98.9% building overlap** confirms the model learns stable building characteristics
- ❌ **Group split fails** — The model cannot generalize to entirely new sites (negative R²)

**Practical Implication:** The model is suitable for:
- Identifying underperformers in an existing building portfolio
- Tracking performance changes over time for known buildings
- NOT suitable for predicting EUI of entirely new buildings/sites

---

## Limitations

The model explains ~70% of log EUI variance, leaving 30% unexplained by factors not captured (occupancy patterns, operational schedules, equipment age, building envelope quality). The geographic concentration of savings in Westminster may reflect genuine inefficiency patterns or dataset composition. This analysis identifies buildings with savings potential but does not estimate retrofit costs, implementation feasibility, or payback periods. All results are derived from aggregated annual energy data for 2016-2017.

Retrofit costs are not estimated because the BDG2 dataset does not include information on upgrade measures, financial costs, or local pricing—only energy, building, and weather data are available.

**Regional Variation:** While ANOVA testing did not find statistically significant differences across regions (p = 0.37), practical concerns remain. Princeton and Westminster show ~150-160% higher RMSE than average. Zoutkamp shows statistically significant systematic under-prediction (t-test p = 0.03). See `report/final_report.md` for a full ethical reflection on fairness implications.

**Generalizability:** While the model performs well on known buildings across years (temporal validation), it cannot reliably predict EUI for entirely new sites. This is because the dataset contains only 18 unique sites, and the model learns site-specific patterns that don't transfer to unseen locations.


---

## Author

Silke Bumann

---

## License

This project uses the BDG2 dataset, which is publicly available for research purposes.
EOF
