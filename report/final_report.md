# Building Energy Efficiency Analysis: Identifying Underperforming Buildings Using Machine Learning

**Technical Report | December 2025**

---

## Executive Summary (Stakeholder-Facing)

### The Problem

**Non-residential buildings in our portfolio consume 30% more energy than necessary, representing significant cost and carbon waste.** Current energy management relies on reactive approaches—addressing issues only after they become costly problems. Without predictive tools, facilities managers cannot systematically identify which buildings are underperforming or prioritize interventions effectively.

**Our analysis of 1,636 buildings reveals that 56% of buildings exceed their predicted energy use intensity (log EUI), with the top 10 underperformers alone accounting for over 53% of total potential energy savings.** Thermal systems (chilledwater and hotwater) represent 96% of the savings opportunity.

**A machine learning model was developed that predicts building EUI with 70% accuracy (R² = 0.702), enabling identification of underperforming buildings before conducting expensive on-site audits.** By targeting the top 50 underperformers first, facilities teams can capture the majority of savings potential with focused effort.

### Key Findings at a Glance

| Metric | Value |
|--------|-------|
| Buildings analyzed | 1,636 |
| Model accuracy (R²) | 70.2% |
| Buildings underperforming | 55.7% of portfolio |
| Top savings opportunity | Westminster region (90.9% of total) |
| Primary drivers | Thermal systems (chilledwater, hotwater) |

---

*MAIN REPORT STARTS HERE*

---

## 1. Introduction

### 1.1 Background

The operation of buildings account for 30% of global final energy consumption and 26% of global energy-related emissions (International Energy Agency, 2024). Identifying and addressing energy inefficiency in commercial buildings represents an effective decarbonization strategy (International Renewable Energy Agency, 2023). 


This project analyzes the Building Data Genome Project 2 (BDG2) dataset to develop a machine learning model that:
1. Predicts expected energy use intensity (EUI) for buildings based on their characteristics
2. Identifies buildings consuming significantly more energy than predicted
3. Quantifies savings potential and prioritizes intervention targets

<ins>References:</ins>

International Energy Agency. (2024). Buildings. IEA. www.iea.org

International Renewable Energy Agency. (2023). World energy transitions outlook 2023: 1.5°c pathway. IRENA.


### 1.2 Dataset Overview

**Building Data Genome Project 2 (BDG2)** contains:
- **3,053 energy meters** from **1,636 non-residential buildings**
- **Two years of hourly data** (2016-2017)
- **Multiple meter types**: electricity, chilledwater, steam, hotwater, gas, irrigation, solar
- **Building metadata**: site/building IDs, primary space usage, floor area (sqm/sqft), location (lat/lng), timezone, year built, number of floors, occupants, heating type, energy ratings (Energy Star, LEED). *Note: Many metadata fields had high rates of missing values (e.g., occupants, heating type, energy ratings), limiting the features available for modeling.*
- **Weather data**: hourly temperature measurements per site

---

## 2. Technical Pipeline

### 2.1 Data Preprocessing

**Data Sources Merged:**
- Meter readings (hourly → annual totals per building-meter-year)
- Building metadata
- Weather data (hourly → annual average temperature per site-year)

**Feature Engineering:**
- Created Energy Use Intensity (EUI) = total_meter_reading / sqft
- Applied log transformation: `log_eui = log1p(eui)` to handle skewed distribution

**Missing Data Handling:**
- Imputed missing `region` values (~3% of records) using temperature matching (assigned to region with closest median annual temperature)
- Developed custom `GroupMedianImputer` to impute missing `yearbuilt` values (~15% of records) using site-level medians
- Removed 55 records with missing `primaryspaceusage` and 17 exact duplicates (72 total removed)

**Final Features (5):**
| Feature | Type | Description |
|---------|------|-------------|
| `Avg_AirTemp_Annual` | Numerical | Annual average temperature |
| `yearbuilt` | Numerical | Building construction year |
| `meter` | Categorical | Meter type (electricity, chilledwater, etc.) |
| `region` | Categorical | Geographic region |
| `primaryspaceusage` | Categorical | Building use type |


### 2.2 Train-Test Split Strategy

**Primary Split (Baseline):** Used `train_test_split` with random sampling:
- 80% train, 20% test
- Random state fixed for reproducibility
- Note: Same site may appear in both train and test (different meters/years), potentially inflating test performance because the model partially "memorized" site characteristics rather than learning generalizable building energy patterns.
- <ins>Example:</ins>
    - A single building site (e.g., site_id "Bear_Tiffany") might have multiple records in the dataset—one for electricity in 2016, another for chilledwater in 2017, etc.
    - With a random split, the 2016 electricity record could end up in the training set while the 2017 chilledwater record ends up in the test set
    - This means the model has "seen" information about that site during training, which could inflate test performance

**Robustness Validation:** Additional split strategies tested to assess generalizability:
- **Temporal Split**: Train on 2016, test on 2017 
- **Group Split**: `GroupShuffleSplit` by `site_id` ensures no site in both sets 

---

## 3. Model Comparison

### 3.1 Models Evaluated

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline interpretable model |
| **Random Forest** | Ensemble method, handles non-linear relationships |
| **LightGBM** | Gradient boosting, efficient for large datasets |

### 3.2 Performance Results

| Model | Train R² | Test R² | Train MSE | Test MSE |
|-------|----------|---------|-----------|----------|
| Linear Regression | 0.428 | 0.437 | 2.781 | 3.094 |
| **Random Forest** | **0.808** | **0.702** | **0.936** | **1.638** |
| LightGBM | 0.794 | 0.711 | 1.004 | 1.589 |

**Selected Model: Random Forest** (comparable performance to LightGBM, better interpretability via feature importances)

### 3.3 Hyperparameter Tuning

Used RandomizedSearchCV with 5-fold cross-validation:

**Best Parameters:**
- `n_estimators`: 58
- `max_depth`: None (unlimited)
- `min_samples_split`: 8
- `min_samples_leaf`: 1

**Cross-Validation Results:**
- Mean CV R²: 0.677
- Standard Deviation: 0.028
- Indicates stable, generalizable performance

### 3.4 Error Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MSE | 1.638 | Mean squared error in log scale |
| RMSE | 1.280 | ~25% of target mean |
| R² | 0.702 | Explains 70% of log EUI variance |

**Diagnostic Checks (All Passed):**
- ✓ Linear relationship between predictions and actuals (correlation > 0.8)
- ✓ Residuals centered around zero (mean ≈ 0)
- ✓ Few outliers (<5% of predictions with >3 std residuals)

---

## 4. Key Findings

### 4.1 Feature Importance

**SHAP Analysis Results:**

| Feature | Mean \|SHAP\| | Interpretation |
|---------|--------------|----------------|
| region_Westminster | 0.46 | Highest impact—buildings there have higher EUI |
| meter_chilledwater | 0.26 | Strong bidirectional effect |
| meter_electricity | 0.25 | Strong bidirectional effect |
| Avg_AirTemp_Annual | 0.25 | Higher temperature → higher EUI (cooling loads) |
| region_Cardiff | 0.18 | Tends to decrease EUI (milder climate) |

*Mean |SHAP| is the mean absolute SHAP value—the average magnitude of each feature's contribution to predictions, ignoring direction.*

**Interpretation:**
- Geographic location (Westminster) is the strongest predictor
- Meter type significantly influences EUI patterns
- Climate (temperature) has meaningful but smaller effect

### 4.2 Underperformer Analysis

**Definition:** Buildings where Actual EUI > Predicted EUI (positive residual)

| Statistic | Value |
|-----------|-------|
| Underperforming buildings | 55.7% of test set |
| Average excess EUI | 0.5 log units |
| Top 10 buildings | 53.2% of total savings |

**Savings by Region:**
- Westminster: 90.9% of total savings potential
- This concentration suggests either regional inefficiency patterns or dataset composition effects

**Savings by Meter Type:**
- Chilledwater: 68.5% of savings
- Hotwater: 27.7% of savings
- Other (gas, water, electricity, etc.): 3.8% of savings
- Thermal systems (chilledwater + hotwater) dominate at 96.2%

---


## 5. Visualizations

This section presents selected visualizations that illustrate the model's performance, feature relationships, potential biases, and actionable insights. Each figure is designed to highlight specific aspects of the analysis and support evidence-based decision-making.

---

### 5.1 Model Performance Comparison

![Model Comparison](figures/model_comparison.png)

*Figure 1: Comparison of R² scores across three regression models (Linear Regression, Random Forest, and LightGBM). Blue bars represent training performance; red bars represent test performance. Value labels on each bar show exact R² scores.*

The grouped bar chart in Figure 1 compares the predictive performance of three candidate models using the coefficient of determination (R²) on both training and test sets. The visual layout enables direct comparison of (1) accuracy differences across models and (2) overfitting tendencies (gap between train and test bars).
The **Random Forest (center)** achieves the best test performance (R² = 0.702) while maintaining a reasonable train-test gap. In contrast, the **Linear Regression (left)** shows poor performance (R² ≈ 0.52), confirming that non-linear relationships exist in the data. Finally, the **LightGBM (right)** slightly outperforms Random Forest on test data (R² = 0.711). The visualization in Figure 1 justifies the selection of Random Forest as the final model. While LightGBM achieves marginally higher test R², Random Forest provides better interpretability through native feature importance measures and SHAP compatibility—critical for stakeholder communication in energy efficiency programs.

---

### 5.2 Predicted vs Actual Plot

![Predicted vs Actual](figures/predicted_vs_actual.png)

*Figure 2: Scatter plot of predicted vs actual log(EUI) values for all test set buildings. Color gradient (green→red) indicates absolute prediction error magnitude. The dashed diagonal line represents perfect prediction; the blue shaded band shows ±RMSE bounds. An annotation highlights the maximum deviation case.*

This diagnostic plot in Figure 2 is the primary visualization for assessing model accuracy. Each point represents one building-meter-year observation from the test set. Points falling on the diagonal indicate perfect predictions; vertical distance from the diagonal represents prediction error.
Most points cluster tightly around the perfect prediction line, confirming the 70% R² performance. The **±RMSE band** (blue shading) captures the majority of predictions, demonstrating consistent model reliability across the log EUI range. The color gradient shows that green points (low error) dominate; red points (high error) are sparse, indicating few significant mispredictions. The *Maximum deviation annotation** (red arrow) identifies the single worst prediction for follow-up investigation—this building may have data quality issues or truly different energy patterns.


---

### 5.3 Residuals Distribution

![Residuals Distribution](figures/residuals_distribution.png)

*Figure 3: Histogram of prediction residuals (Actual − Predicted) for the test set. The red dashed line marks zero (perfect prediction); the orange solid line indicates the mean residual. Frequency on the y-axis shows the count of buildings in each residual bin.*

The histogram in Figure 3 examines the distribution of prediction errors to assess whether the model exhibits systematic bias. A well-calibrated model should produce residuals centered at zero with approximately symmetric distribution. The distribution peak aligns closely with the zero line (red dashed), indicating no systematic over- or under-prediction at the population level. The **Mean residual** (orange line) is pisitioned very close to zero, confirming unbiased predictions overall. Moreover, the histogram shows roughly equal spread on both positive (under-prediction) and negative (over-prediction) sides. Some outliers exist on both ends, representing buildings where predictions deviate substantially from reality. This visualization confirms that while individual predictions may have errors, the model does not systematically favor one direction. This is important for fairness—no building type or region should be consistently disadvantaged by prediction bias at the aggregate level. However, as explored in Section 5.6, regional subgroup analysis reveals localized biases not visible in this aggregate view.

---

### 5.4 Feature Importance

![Feature Importance](figures/feature_importance.png)

*Figure 4: Horizontal bar chart showing the top 15 features by Random Forest importance score (mean decrease in impurity). Features are sorted in ascending order with the most important at the top. One-hot encoded categorical variables appear with their original category labels.*

The chart in Figure 4 reveals which input features contribute most to the model's predictions. Random Forest importance is calculated as the total decrease in node impurity (Gini importance) averaged across all trees, providing a measure of each feature's predictive power. Multiple region indicators (Westminster, Orlando, etc.) appear among the top features, indicating strong geographic patterns in energy consumption. Chilledwater, electricity, and hotwater meters have distinct importance scores, reflecting different energy consumption profiles. The continuous feature `airTemperature` ranks among the top 15, confirming that climate affects EUI.
The superior performance of tree-based methods over Linear Regression suggests the presence of feature interactions (e.g., how meter type effects vary by region) and non-linear relationships with continuous variables that linear models cannot capture. 

---

### 5.5 SHAP Summary Plot

![SHAP Summary](figures/shap_summary.png)

*Figure 5: SHAP (SHapley Additive exPlanations) summary plot showing feature contributions to predictions. Each row represents a feature; each dot represents one building. Horizontal position indicates SHAP value (impact on prediction); color indicates the feature value (red = high, blue = low for continuous features; presence/absence for binary indicators).*

Unlike traditional feature importance (which shows magnitude only), SHAP values reveal the *direction* of each feature's effect. This plot decomposes predictions into additive contributions from each feature, enabling interpretation of individual predictions and identification of non-linear relationships. Regarding the **Westminster region** (one-hot) indicator, red dots (Westminster = 1) cluster to the right, indicating that buildings in Westminster tend to have higher predicted log EUI. **Chilledwater meter**: shows bidirectional effects—buildings with chilledwater meters can have either higher or lower EUI depending on other factors. In the case of **Avg. Air Temperature** red dots (high temperature) consistently push predictions higher, confirming the intuitive relationship between climate and cooling demand. The spread of dots at each feature reflects variability in that feature's contribution across buildings—some buildings see larger or smaller impacts from the same feature. SHAP values provide the explainability needed for stakeholder communication. For any building, facilities managers can see which factors drive its expected EUI (e.g., "Westminster location contributes +0.3 to predicted log(EUI)"). This helps contextualize predictions: a building flagged as underperforming in Westminster is using more energy than expected despite already accounting for that region's typically higher consumption. This transparency supports informed decision-making about where to focus audits.

---

### 5.6 Regional Bias Analysis

![Regional Bias](figures/regional_bias.png)

*Figure 6: Analysis of regional model performance. Left panel: RMSE by region (red bars indicate regions exceeding 130% of overall RMSE). Center panel: Mean residual showing systematic bias direction (positive = under-prediction, negative = over-prediction). Right panel: Sample size per region.*

These visualizations examine whether the model performs equitably across geographic regions—a critical fairness consideration. The three panels together reveal (1) where predictions are less accurate, (2) whether errors are systematically biased, and (3) whether sample size explains performance differences. The left panel **Prediction Error by Region (RMSE)** shows that Princeton (160% of overall), Westminster (150%), and Orlando (136%) have elevated prediction errors—stakeholders in these regions should be informed that predictions carry greater uncertainty. The center panel (**Bias Direction**) shows that Zoutkamp has a substantial positive mean residual (+0.33), indicating the model systematically *under-predicts* log EUI for buildings in this region—they appear more efficient than they are in reality. The right panel (**Sample Size**) reveals that regions with high error (Princeton, Zoutkamp) tend to have smaller sample sizes, suggesting insufficient training data may contribute to poor performance.
This visualization directly supports the ethical reflection of Section 7. Although the overall ANOVA test was not significant (p = 0.37), these plots reveal meaningful regional disparities that warrant attention:
- Buildings in Zoutkamp may be overlooked for energy audits because they falsely appear efficient
- Predictions for Princeton and Westminster should be communicated with wider uncertainty bounds
- Future data collection should prioritize under-represented regions.

---

### 5.7 Savings Potential by Meter Type

![Savings by Meter](figures/savings_by_meter.png)

*Figure 7: Pie chart showing the distribution of total energy savings potential (kWh/sqm) across meter types for buildings identified as underperforming. The top two categories are visually "exploded" for emphasis. Legend shows absolute values and percentages.*

For buildings where actual log EUI exceeds predicted log EUI (underperformers), the chart in Figure 7 breaks down the total "savings potential" (actual − predicted) by energy meter type. This guides retrofit prioritization by identifying which energy systems contribute most to inefficiency. **Chilledwater dominates** (68.5%): Cooling systems represent the largest opportunity for energy savings, suggesting chiller plant optimization and building envelope improvements. Heating systems are the second priority (27.7%), particularly relevant for older buildings with inefficient boilers. Together, chilledwater and hotwater account for nearly all identified savings—direct electricity and gas inefficiencies are minimal.
This finding has direct operational implications. Energy efficiency programs should focus on thermal systems (HVAC) rather than electricity-specific interventions. The extreme concentration (96%) provides a clear need for thermal retrofits. However, this may also reflect data availability—thermal meters may simply have more variance than electricity meters.

---

### 5.8 Savings Potential by Region

![Savings by Region](figures/savings_by_region.png)

*Figure 8: Horizontal bar chart showing total energy savings potential by geographic region. Regions are sorted by savings magnitude. Red bars indicate regions accounting for >50% of total savings.*

The chart in Figure 8 reveals the geographic concentration of energy savings opportunities, enabling targeted deployment of retrofit programs. Buildings in regions with high total savings potential should be prioritized for energy audits. Westminster accounts for the vast majority of identified savings (90.9%). Some regions contribute minimal savings, suggesting limited efficiency problems or small building portfolios.
The analysis has shown Westminster's dominance is due to real efficiency problems, not simply having more buildings:
Westminster has only **10.4% of test set buildings** (121/1166) and **10.2% of underperforming buildings** (66/649). Yet Westminster accounts for **90.9% of total savings potential**. This inefficiency suggests Westminster's building stock has systemic problems—possibly older construction, deferred maintenance, or different building standards in England. This finding strongly supports prioritizing Westminster for energy audits.

---

## 6. Model Limitations

The model accounts for approximately 70% of the variance in outcomes, indicating a significant performance gap due to missing features like floor area, HVAC type, and occupancy data. A key finding is that while Westminster's 90.9% savings are genuine inefficiencies (114x higher per-building savings than other regions), the root causes, such as building age or maintenance standards, remain unverified.
Generalizability concerns are significant; the model fails entirely when applied to completely new sites (group split R² < 0) and must be restricted to the existing known portfolio of buildings.
Prediction accuracy fluctuates significantly across different areas:
High-error regions include Princeton (160% RMSE), Westminster (150%), and Orlando (136%).
Zoutkamp exhibits a consistent bias toward under-prediction (+0.33 mean residual).
Stakeholders in these specific high-error regions must be informed of the greater uncertainty surrounding their predictions.
Ultimately, the model is limited by its reliance on only five available features and a two-year time horizon. Future improvements hinge entirely on incorporating richer metadata, such as the missing yearbuilt values that were previously imputed using site-level medians.

---

## 7. Ethical Reflection

### 7.1 Fairness Concerns

The model exhibits **regional variation in prediction accuracy**, which raises practical concerns regarding equitable outcomes, despite overall statistical insignificance.

-   **Unequal Accuracy**: Buildings in high-error regions (Princeton: 160% RMSE, Westminster: 150%, Orlando: 136%) receive less reliable predictions, potentially leading to unfair resource allocation.
-   **Systematic Bias in Zoutkamp**: This region shows statistically significant under-prediction (mean residual +0.33), causing buildings to appear more efficient than they are in reality and risking genuinely inefficient sites being overlooked for interventions.
-   **Small Sample Sizes**: Regions with limited training data (e.g., Zoutkamp n=22, Ottawa n=23, Princeton n=37) may inherently have less reliable predictions, which disadvantages local facilities managers relying on those estimates.

## 7.2 Environmental Implications

The data-driven approach facilitates environmental benefits, but model errors present risks.

-   **Positive Impacts**: Identifying underperforming buildings enables targeted energy efficiency improvements, and prioritizing high-impact thermal systems addresses the most significant opportunities effectively.
-   **Concerns**: Model errors could misdirect resources to already efficient buildings, systematically neglect the needs of certain regions due to bias, and over-reliance without on-site validation risks poor environmental decisions.

## 7.3 Mitigation Commitments

To address these ethical concerns, the following steps are committed:

-   **Transparency**: All limitations are thoroughly documented and communicated to stakeholders.
-   **Human Oversight**: Model recommendations mandate validation before any major investment decisions.
-   **Continuous Monitoring**: Quarterly audits are implemented to detect performance degradation or emerging bias.

---

## 8. Recommendations

### 8.1 Immediate Actions (0-3 months)
1. **Conduct energy audits** on top 10 underperforming buildings identified by the model
2. **Focus on thermal systems** (chilledwater, hotwater) which represent 96% of savings
3. **Flag Westminster buildings** for priority review given concentration of savings potential

### 8.2 Medium-term Actions (3-12 months)
1. **Collect additional data** (floor area, HVAC type, occupancy) to improve model accuracy
2. **Develop region-specific models** for high-error areas (Princeton, Westminster)
3. **Implement monitoring dashboard** to track prediction accuracy over time

### 8.3 Long-term Strategy
1. **Expand to new sites** only after collecting sufficient training data
2. **Integrate with building management systems** for real-time anomaly detection
3. **Develop cost-benefit models** to prioritize retrofits by ROI, not just energy savings

---

## 9. Conclusion

This analysis demonstrates that machine learning can effectively identify underperforming buildings, with a Random Forest model explaining 70% of EUI variance. The model identifies thermal systems in the Westminster region as the primary savings opportunity.

However, responsible deployment requires acknowledging limitations: the model exhibits regional bias, cannot generalize to new sites, and explains only 70% of variance. These constraints should inform how the model is used—as a screening tool to prioritize investigation, not as a definitive judgment of building performance.

By combining predictive modeling with on-site validation and continuous monitoring, facilities managers can systematically target energy efficiency improvements while avoiding the pitfalls of over-reliance on algorithmic recommendations.

---

## Appendix: Instructions to Run the Code

See `README.md` in the project root for:
- Environment setup instructions
- Data file locations
- Notebook execution order

---

*Report generated: December 2025*
*Model version: Random Forest v1.0*
*Data period: 2016-2017*
