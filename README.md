# Decarbonizing the Built Environment: Identifying High-Impact Opportunities within the Building Data Genome Project 2 (BDG2)

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
* The project report is in ...
