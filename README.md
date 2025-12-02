<!-- # Decarbonizing the Built Environment: Identifying High-Impact Opportunities within the Building Data Genome Project 2 (BDG2)

* * * * *

## The Context: Staying Within the 1.5°C Carbon Budget

The recent Emissions Gap Report underscores a critical global challenge: to have a high likelihood (more than 80%) of staying within the 1.5∘C warming threshold, the world has a remaining carbon budget of approximately 300 gigatonnes of global 
CO<sub>2</sub> emissions.

The built environment (buildings sector) is a major contributor to global energy consumption and associated CO<sub>2</sub> emissions, offering substantial potential for immediate and scalable reductions. This project employs ML regression techniques to pinpoint specific inefficiencies within real-world building data, thereby providing actionable insights.

## Project Goal

The primary goal of this project is to analyze the energy consumption data of non-residential buildings provided by the BDG2 dataset to:

1.  Establish a predictive baseline for building energy consumption.
2.  Identify buildings that are significantly underperforming compared to similar buildngs. <!-- and climate conditions. -->
3. Identify the drivers of CO<sub>2</sub> emissions in buildings.
<!-- 4.  Quantify the potential CO<sub>2</sub> emission reductions achievable through operational improvements and retrofits in these high-impact buildings. -->

## Dataset

**Building Data Genome Project 2 (BDG2)**

BDG2 is an open data set made up of 3,053 energy meters from 1,636 buildings. The time range of the times-series data is two full years from 2016 to 2017 and the frequency is hourly measurements of electricity, heating and cooling water, steam, and irrigation meters.

The BDGE dataset consists of three primary types of data:

**Building Metadata:** Contains information about the buildings, including their types, locations, and specific characteristics that may influence energy consumption (e.g., floor area, building age, and facility type).

**Meter Data:** Includes readings from various energy meters installed in the buildings, capturing consumption data for different types of energy sources - electricity, chilled water, steam, hot water, and gas. The meter data is detailed, showing usage over time, typically recorded on an hourly basis.

**Weather Information:** Provides local weather data relevant to the buildings' locations. This part of the dataset includes temperature, humidity, wind speed, and other weather-related variables that can significantly affect energy consumption patterns.

**Dataset Source:** GitHub Repository: https://github.com/mahishah19/Energy-Prediction-Using-BDG2-Data/blob/main/README.md

## Methodology

A simplified, cross-sectional regression approach is employed, rather than more complex time-series analysis:

1.  **Data Aggregation:** Hourly time-series data is aggregated to annual summary statistics (e.g., total annual kWh consumption, total heating/cooling degree days). This transforms the data into a single observation per building per year.
2.  **Features:** Key predictors like building floor area, age, type, climate zone, and aggregated weather metrics are used as independent variables.
    * Aggregate hourly weather data into annual summaries like Total_Heating_Degree_Days (HDD) and Total_Cooling_Degree_Days (CDD).
3.  **Regression Modeling:** A robust regression model (e.g., Linear Regression and Random Forest) is trained to predict the "expected" energy use based on these features.
4. **Efficiency Analysis:** Compare actual consumption to the model's predicted consumption to identify buildings with significant excess usage, indicating high potential for CO2 reduction.
<!-- 5.  **Anomaly Detection & Quantification:** Buildings consuming significantly more energy than their predicted baseline are flagged as high-potential candidates for efficiency improvements. The difference is used to calculate potential energy savings.
6. **CO<sub>2</sub>Calculation:** Regional emission factors are applied to the quantified energy savings to determine actual tons of CO<sub>2</sub> reduction potential. -->

<!-- ## Project Structure

The repository is structured to produce a comprehensive final report and supporting code:

| Folder | Description |
| ------- | ----------- |
| `jupyter_notebooks/` | Jupyter notebooks for data cleaning, aggregation, modeling, and analysis. |
| `data/` | Contains the raw and cleaned BDG2 data files |
| `src/` | Helper scripts for data processing and model evaluation. |
| `report/` | Final project report and presentation files. |
 -->


## Repository structure

    .
    ├── jupyter_notebooks/
    │   ├── preprocessing/
    │   └── modelling/
    ├── data/
    │   ├── meters/
    │   │   ├── raw/
    │   │   └── cleaned/
    │   ├── metadata/
    │   └── weather/
    ├── report/
    └── README.md




## Key Technologies & Libraries

-   Python
-   Pandas (Data manipulation and aggregation)
-   Scikit-learn (Modeling and validation)
-   Matplotlib/Seaborn (Visualization)


## Getting Started

* The data cleaning steps can be found in ...
* The modelling part can be found in ...
* The project report is in ... -->





# Decarbonizing the Built Environment: Identifying High-Impact Opportunities within the Building Data Genome Project 2 (BDG2)

---

## The Context: Staying Within the 1.5°C Carbon Budget

The recent Emissions Gap Report underscores a critical global challenge: to have a high likelihood (more than 80%) of staying within the 1.5°C warming threshold, the world has a remaining carbon budget of approximately 300 gigatonnes of global CO₂ emissions.

The built environment (buildings sector) is a major contributor to global energy consumption and associated CO₂ emissions, offering substantial potential for immediate and scalable reductions. This project employs ML regression techniques to pinpoint specific inefficiencies within real-world building data, thereby providing actionable insights for targeted carbon reduction strategies.

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
| Linear Regression | 0.43 | 0.40 |
| Random Forest (default) | 0.84 | 0.67 |
| **Tuned Random Forest** | **0.82** | **0.68** |
| Tuned LightGBM | 0.82 | 0.68 |

The tuned Random Forest model explains **68% of EUI variance**, validated by comparable LightGBM performance.

### Savings Potential Analysis

From 1,170 test buildings, **614 underperformers (52.5%)** were identified with a total savings potential of **5.37 million kWh/sqm**.

| Breakdown | Savings (kWh/sqm) | % of Total |
|-----------|-------------------|------------|
| **Chilledwater (cooling)** | 4,341,318 | **80.8%** |
| **Hotwater (heating)** | 820,504 | **15.3%** |
| Electricity (direct) | 39,811 | 0.7% |
| Gas (Scope 1) | 77,021 | 1.4% |

### Key Insight

**Thermal systems (chilledwater + hotwater) account for 96% of all savings potential.** Direct electricity and gas inefficiency is minimal.

### Priority Targets

- **Region**: City of Westminster, England — 92.5% of all savings potential
- **Building Type**: Education — 64.1% of total savings (256 underperforming buildings)
- **Highest per-building inefficiency**: Healthcare — ~25,922 kWh/sqm average per building

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

1. **Data Aggregation:** Hourly time-series data aggregated to annual summary statistics
2. **Features Used:**
   - Categorical: `primaryspaceusage`, `region`, `meter`
   - Numerical: `yearbuilt`, `Avg_AirTemp_Annual`
3. **Missing Data Handling:**
   - `region`: Temperature-based imputation (before train/test split)
   - `yearbuilt`: Group-wise median imputation by site (after split to prevent leakage)
4. **Modeling:** Random Forest with hyperparameter tuning via RandomizedSearchCV (5-fold CV)
5. **Validation:** LightGBM comparison to confirm robustness
6. **Efficiency Analysis:** Buildings with positive residuals (actual > predicted) flagged as underperformers

---

## Repository Structure


    .
    ├── jupyter_notebooks/
    │   ├── meta_data_preprocessing.ipynb      # Metadata cleaning
    │   ├── meter_data_preprocessing.ipynb     # Raw meter data processing
    │   ├── meter_data_cleaned_preprocessing.ipynb  # Cleaned meter aggregation
    │   ├── weather_preprocessing.ipynb        # Weather data processing
    │   ├── EDA.ipynb                          # Exploratory data analysis
    │   └── modeling_cleaned.ipynb             # Main modeling notebook ⭐
    ├── data/
    │   ├── meters/
    │   │   ├── raw/                           # Original meter CSVs
    │   │   └── cleaned/                       # Processed meter data
    │   ├── metadata/                          # Building metadata
    │   └── weather_data/                      # Weather information
    ├── lecture_notes/                         # Reference materials
    └── README.md


---

## Key Technologies & Libraries

- **Python 3.x**
- **Pandas** — Data manipulation and aggregation
- **NumPy** — Numerical operations
- **Scikit-learn** — Modeling, preprocessing, cross-validation
- **LightGBM** — Gradient boosting for model validation
- **Matplotlib/Seaborn** — Visualization

---

## Getting Started

### Prerequisites

\`\`\`bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
\`\`\`

### Running the Analysis

1. **Data Preprocessing:** Run the preprocessing notebooks in order:
   - `meta_data_preprocessing.ipynb`
   - `meter_data_preprocessing.ipynb`
   - `meter_data_cleaned_preprocessing.ipynb`
   - `weather_preprocessing.ipynb`

2. **Modeling:** Open and run `modeling_cleaned.ipynb` — this is the main analysis notebook containing:
   - Model training and hyperparameter tuning
   - Feature importance analysis
   - Savings potential calculations by building type, region, and meter
   - Top 10 priority buildings for retrofit
   - Limitations and recommendations

---

## Recommendations

Based on the analysis, the following actions are recommended for maximum CO₂ reduction impact:

1. **Prioritize HVAC retrofits** — 96% of savings comes from thermal systems (chilledwater + hotwater)
2. **Focus on Westminster** — 92.5% of savings concentrated in one region
3. **Target Education buildings at scale** — 256 underperformers representing 64% of total savings
4. **Audit Healthcare buildings for quick wins** — Fewer buildings but highest per-building impact
5. **Conduct energy audits on top 10 buildings** — Captures ~40% of total savings potential

---

## Limitations

The model explains 68% of EUI variance, leaving 32% unexplained by factors not captured (occupancy patterns, operational schedules, equipment age, building envelope quality). The geographic concentration of savings in Westminster may reflect genuine inefficiency patterns or dataset composition. This analysis identifies where savings exist but does not evaluate retrofit costs or payback periods. Wile the dataset covers two years (2016–2017), performance patterns may shift over longer time horizons.

---

## Author

Silke Bumann

---

## License

This project uses the BDG2 dataset, which is publicly available for research purposes.
EOF
