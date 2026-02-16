# Ecommerce Sales Prediction - Capstone Project

## 📌 Project Overview
This project aims to optimize inventory management and marketing strategies for an Ecommerce platform by analysing historical sales data. The core objective is to predict future product sales volumes and classify demand levels to trigger inventory alerts.

**Research Question:**
> Can future sales volumes of Amazon products be accurately predicted using historical data and market trends to proactively optimize inventory levels?

## 🎯 Objectives
1.  **Sales Velocity Prediction (Regression)**: Build a model to predict the exact number of units sold based on price, marketing spend, and product characteristics.
2.  **Demand Classification (Classification)**: Classify days as "High Demand" or "Low Demand" to assist in inventory planning.

## 📊 Data Understanding & EDA
**Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/nevildhinoja/e-commerce-sales-prediction-dataset)

The dataset consists of **1,000 records** containing daily sales data with features such as `Price`, `Marketing_Spend`, `Product_Category`, and `Customer_Segment`.

### Key Findings
*   **Data Quality**: The dataset is perfectly clean (0 nulls, 0 duplicates).
*   **Correlation Analysis**:
    *   **Price**: Correlation with Sales is **0.015** (Effectively zero).
    *   **Marketing Spend**: Correlation with Sales is **0.009** (Effectively zero).
*   **Verdict**: Traditional linear drivers (Price & Marketing) show **no linear relationship** with Sales in this dataset. This suggests the data may be synthetic or driven by complex non-linear patterns, necessitating advanced models like Random Forest or XGBoost.

### "The Broken Lever"
Our analysis revealed that the standard "business levers" are disconnected:
*   Increasing **Marketing Spend** does *not* proportionally increase sales.
*   Changing **Price** does *not* impact demand (perfect inelasticity).

## 🛠 Methodology (CRISP-DM)
This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:
1.  **Business Understanding**: Defining goals and success criteria.
2.  **Data Understanding**: Exploratory Data Analysis (EDA) using Matplotlib, Seaborn, and Plotly.
3.  **Data Preparation**: Feature Engineering (One-Hot Encoding, Date Extraction) and Scaling.
4.  **Modeling**: Implementing Linear Regression, Ridge, Decision Tree, Random Forest, and XGBoost.
5.  **Evaluation**: Comparing models using RMSE and R-squared metrics.

## 🚀 How to Run
1.  Ensure you have Python installed with the following libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
    ```
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Capstone_Sales_Prediction.ipynb
    ```
    **Link to Notebook**: [Capstone_Sales_Prediction.ipynb](Capstone_Sales_Prediction.ipynb)
3.  Run all cells to reproduce the analysis and model results.

## 📂 Project Structure
*   `Capstone_Sales_Prediction.ipynb`: Main analysis and modeling notebook.
*   `data/`: Directory containing the dataset (`Ecommerce_Sales_Prediction_Dataset.csv`).
*   `README.md`: Project documentation (this file).