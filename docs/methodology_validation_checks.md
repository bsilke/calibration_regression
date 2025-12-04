# Methodology Validation Checks

This document summarizes key methodological considerations for the EUI prediction model, demonstrating awareness of potential issues and the reasoning behind design decisions.

---

## 1. Data Aggregation Timing

**Question:** The original data was hourly meter data, aggregated to yearly data before the train-test split. Is this a problem?

**Answer:** ✅ **No issue.**

Aggregating hourly → yearly is a **feature engineering** step, not a model-fitting step. We are not learning anything from the target variable during aggregation—just summarizing raw measurements. The key concern would be if information from the test set informed the aggregation (e.g., normalizing by the global mean), but simple summation/averaging does not cause leakage.

---

## 2. Preprocessing on Training Set Only

**Question:** Is one-hot encoding and scaling done on the training set only?

**Answer:** ✅ **Correctly implemented.**

The scikit-learn `Pipeline` encapsulates the `ColumnTransformer` (containing `StandardScaler` and `OneHotEncoder`). When `pipeline.fit(X_train, y_train)` is called, preprocessing steps are fitted **only on training data**. The pipeline then correctly applies `transform` (not `fit_transform`) to test data during prediction.

**Minor note:** The LightGBM implementation fits `LabelEncoder` on combined train+test data to ensure all categories are known. This is a minor technical leakage that rarely impacts results for categorical encoding but should be acknowledged.

---

## 3. Target and Preprocessing Leakage Analysis

**Question:** Could the model have target or preprocessing leakage?

**Answer:** ✅ **No significant leakage detected.**

| Component | Status | Reasoning |
|-----------|--------|-----------|
| Target variable (`log_eui`) | ✅ Safe | Pre-computed in dataset, extracted after train-test split, not used in preprocessing |
| `GroupMedianImputer` | ✅ Safe | Fits on training data only, applies learned medians to test |
| `StandardScaler` | ✅ Safe | Inside Pipeline, fitted on train only |
| `OneHotEncoder` | ✅ Safe | Inside Pipeline, fitted on train only |
| LightGBM `LabelEncoder` | ⚠️ Minor | Fits on train+test combined (low impact for categorical encoding) |

---

## 4. Pooling Data Across Years (2016 and 2017)

**Question:** Is pooling data from 2016 and 2017 an issue?

**Answer:** ✅ **Acceptable for benchmarking use case.**

| Approach | Pros | Cons |
|----------|------|------|
| **Pool 2016+2017** (current) | Larger sample size, captures inter-annual variation | Treats same building in different years as independent (violates i.i.d. assumption) |
| Use only one year | Cleaner independence | Loses data, less robust |
| 2016 train → 2017 test | True temporal validation | May capture year-specific effects rather than building patterns |

**Justification:** For portfolio benchmarking (our use case), pooling is reasonable because:
- We're building a cross-sectional model of "what EUI should a building have given its characteristics"
- Year-to-year variation within the same building is typically small compared to between-building variation
- The model could include `year` as a feature to capture any systematic annual effects if needed

---

## 5. Data Splitting Strategy

**Question:** With time-ordered data (2016, 2017) and multiple meters per location, is a random split appropriate?

**Answer:** ⚠️ **Random split is acceptable but has limitations.**

### Data Structure
The dataset has two types of structure:
1. **Temporal:** Same building observed in 2016 and 2017
2. **Hierarchical:** Multiple meters per site (`site_id`)

### Current Approach: Random Row-Level Split
- Same `site_id` can appear in both train and test sets (different meters or years)
- The model may learn site-specific patterns that don't generalize

### Alternative Approaches

**Option A: Group-based split by site_id** (recommended for generalizability claims)
```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['site_id']))
```

**Option B: Temporal split**
```python
train_df = df[df['year'] == 2016]
test_df = df[df['year'] == 2017]
```

### Decision Rationale
For **portfolio benchmarking** (identifying underperformers among known buildings), random split is acceptable because we're applying the model to buildings already in our dataset. However, if claiming the model **generalizes to new buildings**, a group-based split would provide more rigorous validation.

---

## 6. Scaler Selection

**Question:** Is `StandardScaler` the right choice?

**Answer:** ✅ **Appropriate choice.**

| Scaler | When to Use | Our Data |
|--------|-------------|----------|
| **StandardScaler** (current) | General purpose, assumes roughly normal distribution | ✅ Suitable |
| RobustScaler | Data has significant outliers | Not needed |
| MinMaxScaler | Need values in [0,1] range | Not required |

**Justification:**
- Random Forest doesn't require scaling (tree-based), but scaling doesn't hurt
- Linear Regression benefits from scaled features for numerical stability
- Our numerical features (`yearbuilt` ~1900-2017, `Avg_AirTemp_Annual` ~0-30°C) have different scales
- StandardScaler normalizes both to comparable ranges

---

## Summary Table

| Check | Status | Report Mention |
|-------|--------|----------------|
| 1. Yearly aggregation before split | ✅ OK | Brief mention |
| 2. OHE/Scaling on train only | ✅ OK | Yes - demonstrate awareness |
| 3. Target/preprocessing leakage | ✅ OK (minor LightGBM note) | Yes - important for credibility |
| 4. Pooling 2016+2017 | ✅ OK | Yes - justify decision |
| 5. Random vs group split | ⚠️ Limitation | **Yes - key limitation to discuss** |
| 6. StandardScaler choice | ✅ OK | Brief mention |

---

## Recommended Points to Address in Report

### Must Include (demonstrates methodological rigor)

1. **Data Splitting Strategy (Point 5)**
   - Acknowledge that random splitting may allow the same site to appear in train and test
   - Explain why this is acceptable for portfolio benchmarking
   - Note that for true generalizability to new buildings, a group-based split would be preferred
   - This shows awareness of a common ML pitfall

2. **Leakage Prevention (Point 3)**
   - Briefly state that preprocessing (scaling, encoding, imputation) was fitted on training data only
   - Mention the Pipeline approach ensures no test data influences the preprocessing
   - This demonstrates understanding of a critical ML best practice

3. **Year Pooling Decision (Point 4)**
   - Justify why pooling 2016+2017 is appropriate for your analysis goal
   - Acknowledge this treats observations as independent when they may have temporal correlation

### Optional (strengthens the report)

4. **Aggregation Timing (Point 1)**
   - One sentence noting that aggregation is a feature engineering step, not a modeling step

5. **Scaler Choice (Point 6)**
   - Brief mention that StandardScaler was used for numerical features to ensure comparable scales for linear regression

### Sample Report Language

> **Methodological Considerations**
>
> Several steps were taken to prevent data leakage. All preprocessing transformations (StandardScaler, OneHotEncoder, and GroupMedianImputer) were fitted exclusively on the training set and applied to the test set without refitting. This was implemented using scikit-learn's Pipeline API to ensure consistent handling.
>
> The dataset contains observations from both 2016 and 2017, which were pooled to increase sample size. While this treats multiple observations from the same building as independent, this is acceptable for our portfolio benchmarking objective where we compare buildings against expected performance rather than predicting for entirely new buildings.
>
> A random train-test split was used, meaning the same site may appear in both sets (different meters or years). For applications requiring generalization to unseen buildings, a group-based split by `site_id` would provide more rigorous validation. However, since our goal is to identify underperformers within a known portfolio, this approach is appropriate.

---

## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 7: Model Assessment and Selection.
- Kaufman, S., Rosset, S., & Perlich, C. (2011). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*.
- scikit-learn documentation: [Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)
