# 📊 Outlier Treatment in Regression: California Housing Case Study

This project explores the impact of different outlier treatment techniques on the performance of regression models using the **California Housing dataset**. Specifically, we compare:

- IQR-based Outlier Removal
- Log Transformation of the target variable

---

## 🗂️ Dataset Overview

**Source:** 1990 U.S. Census

**Target:** Median house value (in \$100,000s)

**Features:**
- `MedInc` – Median income in block group
- `HouseAge` – Median house age
- `AveRooms` – Average number of rooms
- `AveBedrms` – Average number of bedrooms
- `Population` – Block group population
- `AveOccup` – Average household occupancy
- `Latitude`, `Longitude` – Geographic coordinates

This dataset is commonly used for regression benchmarking in machine learning.

---

## 🧪 Methods Compared

| Method            | Description |
|------------------|-------------|
| **IQR Removal**   | Removes data points with target values beyond 1.5×IQR |
| **Log Transform** | Applies `log1p` transformation on the target to reduce skew |

---

## 📉 Outlier Detection Example (IQR Method)

```python
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
mask = (y_train >= lower) & (y_train <= upper)
