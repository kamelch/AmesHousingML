# House Price Prediction Engine

A machine learning project designed to predict residential property sale prices in Ames, Iowa. This project utilizes the **Ames Housing Dataset** and implements a comprehensive pipeline including data cleaning, advanced feature engineering, outlier detection, and regression modeling.

## 📊 Project Overview

* **Objective:** Predict final sale prices of 500+ homes based on 80+ explanatory variables.
* **Dataset:** Ames Housing Data (2930 observations, 82 features).
* **Author:** Kamel Chahbi.

## 🧠 Key Features & Methodology

The solution follows a rigorous data science workflow:

### 1. Exploratory Data Analysis (EDA)
* **Data Profiling:** Analysis of 80 features including nominal (Zoning, Street), ordinal (Lot Shape, Utilities), discrete (Year Built), and continuous variables (Lot Area).
* **Visualizations:** Correlation heatmaps and distribution plots to understand price factors.

### 2. Advanced Preprocessing
* **Data Cleaning:** Handling missing values for Masonry Veneer, Basements, and Garages by imputing with 'None' or 0 where appropriate.
* **Outlier Detection:** Removal of extreme outliers (e.g., in `Lot Area` or `SalePrice`) using statistical methods (Z-score > 3) to prevent model skew.
* **Encoding:** Transformation of categorical variables for machine learning compatibility.

### 3. Feature Engineering & Selection
* **Selection:** Usage of `SelectKBest` and `mutual_info_regression` to identify the most predictive features.
* **Scaling:** Implementation of `RobustScaler` to handle outliers in feature scaling.

### 4. Modeling
* **Algorithm:** Uses **Ridge Regression** (Linear Regression with L2 Regularization) to handle multicollinearity among features.
* **Iterations:** The notebook compares three model variants and exports their predictions separately:
    * `predictions-simple-model.csv`
    * `predictions-intermediate-model.csv`
    * `predictions-complex-model.csv`

---

## 🛠️ Dependencies

* **Python 3.x**
* `pandas` & `numpy` (Data manipulation)
* `matplotlib` & `seaborn` (Visualization)
* `scikit-learn` (Modeling, metrics, preprocessing)

---

## 🚀 Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kamelch/AmesHousingML.git
    cd AmesHousingML
    ```
2.  **Install requirements:**
    ```bash
    python -m pip install -r requirements.txt
    python -m jupyterlab
    ```
3.  **Run the Notebook:**
    Open `house-prices-solution.ipynb` in Jupyter Notebook or JupyterLab to execute the analysis pipeline.

---

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `house-prices-solution.ipynb` | **Core Notebook.** Contains all EDA, cleaning, and modeling logic. |
| `house-prices.csv` | Raw training dataset. |
| `documentation.txt` | Detailed description of all dataset columns. |
| `predictions-*.csv` | Model output files containing price predictions. |


## Reproducing the notebook

Launch JupyterLab from the repository root so relative CSV paths resolve correctly.
The notebook records a Python 3.8.5 kernel. `requirements.txt` provides compatibility
bounds for its older scientific Python APIs; use a Python 3.10 environment for this
setup. These bounds are not a lockfile or a claim of an end-to-end reproduction.

`house-prices-test.csv` contains prediction inputs, and `predictions-example.csv`
provides the output schema. The simple, intermediate, and complex prediction CSVs
are separate model results, not backup copies. Running the final cells overwrites
those tracked results; inspect the diff before committing them.

Dataset attribution is in [source.txt](source.txt), with column descriptions in
[documentation.txt](documentation.txt). See [CONTRIBUTING.md](CONTRIBUTING.md)
for repository and commit conventions.
