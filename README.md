### Amazon Sales Prediction: A Comprehensive Machine Learning & Deep Learning Analysis

**Author:** Krishnamoorthy Dharmalingam

#### Executive Summary
This project aims to optimize Amazon e-commerce operations by forecasting product-level sales velocity, classifying high-demand events, and generating macroeconomic store-wide daily sales forecasts. By applying the CRISP-DM framework, we utilized a tiered modeling approach—from Baseline Linear and Logistic Regressions to Advanced Ensemble methods (XGBoost) and sophisticated Deep Learning architectures (Bidirectional LSTMs and Softmax Multi-Class Neural Networks). The resulting models provide actionable intelligence for supply chain and inventory management.

#### Rationale
E-commerce inventory management presents a massive financial challenge. Overstocking unmoving inventory results in devastating FBA (Fulfillment by Amazon) long-term storage fees, while stockouts of high-velocity items sacrifice primary revenue and heavily damage algorithmic product ranking. Developing highly accurate predictive pipelines can optimize logistics, streamline ad-spend budgeting, and drastically boost overall profitability via efficient capital allocation.

#### Research Question
1. Can we accurately forecast the specific numeric demand (`Units_Sold`) of a product on any given day?
2. Can we proactively classify whether a product will experience a "High Demand" spike to prepare logistics?
3. Can a Deep Learning architecture interpret historical chronological 7-day windows to accurately forecast macroscopic, store-wide daily sales volume?
4. Is it possible to categorize an item's macro `Product_Category` purely by analyzing numerical metrics like `Price`, `Ad_Spend_PPC`, and `Stock_Level`?

#### Data Sources
What data will you use to answer your question?

The primary dataset utilized is `amazon_sales_dataset.csv` (sourced from [Kaggle](https://www.kaggle.com/)), consisting of 10,000 recorded Amazon transactions across the year 2023. Key predictive features include financial metrics (`Price`, `Discount_Percent`), marketing spend (`Ad_Spend_PPC`), marketplace dominance (`Amazon_Buy_Box_Percentage`), logistics strategy (`FBA_Status`), and external/temporal elements (`Day_Of_Week`, `Is_Holiday`, `Weather`, `Season`).

#### Methodology
The project strictly adhered to the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:
*   **Data Preparation:** Handled missing values, parsed DateTime objects, encoded categorical variables via `OneHotEncoder` and `LabelEncoder`, and normalized numerical columns using `StandardScaler` and `MinMaxScaler`.
*   **Regression (Micro Demand):** Evaluated Baseline Linear Regression against complex ensembles including Random Forest Regressor and XGBoost Regressor (measured via RMSE).
*   **Classification (Demand Spikes):** Evaluated Baseline Logistic Regression against a Support Vector Classifier utilizing an RBF Kernel (measured via F1-Score).
*   **Deep Learning (Macro Time-Series):** Aggregated data chronologically to train a stacked **Bidirectional LSTM** Neural Network, featuring Dropout regularization and Relu/Sigmoid dense layers, to interpret sequentially dependent market movement.
*   **Deep Learning (Multi-Class):** Developed a deep Dense Neural Network topped with a **Softmax** activation layer and Sparse Categorical Crossentropy to classify strings into product categories.

#### Results
*   **Inventory Replenishment (Regression):** To prevent stockouts and FBA overstock fees, the XGBoost Regressor forecasts daily `Units_Sold` per product (RMSE: ~31.21), achieving comparable RMSE to the Linear Regression baseline (~30.47) while offering superior generalization by capturing non-linear promotional spikes driven by Pay Per Click (PPC) ad spending and holidays.
*   **Logistics Readiness (Classification):** To proactively prepare warehouse and fulfillment capacity, the SVC Classifier identifies "High Demand" spike days with an F1-Score of ~0.80, achieving comparable performance to the Logistic Regression baseline (~0.82) while demonstrating greater robustness in navigating high-dimensional feature spaces via an RBF kernel.
*   **Strategic Purchasing (LSTM Time-Series):** To guide bulk inventory procurement decisions, the Bidirectional LSTM interprets 7-day chronological windows to forecast **store-wide daily aggregate sales trajectories** (RMSE: ~52.56 on a large-scale aggregate volume), serving as a precise macro-level indicator for anticipating demand curves ahead of supplier negotiations.
*   **Catalog Intelligence (Multi-Class):** To automate product categorization for ad targeting and pricing strategy, the Softmax Neural Network classifies `Product_Category` from the full product feature set, validated via Seaborn Confusion Matrices and Classification Reports confirming strong probabilistic separation across classes.

#### Next Steps
1.  **API Deployment:** Wrap and deploy the final aligned XGBoost Regressor and Keras LSTM models as a REST API (using FastAPI). ✅ Done
2.  **Docker Containerization:** Package the entire API service into a Docker container. ✅ Done
3.  **Automated Purchasing:** Programmatically tie the LSTM 30-day macro forecast directly into wholesale purchasing systems to negotiate bulk inventory discounts ahead of demand curves.
4.  **Monitor Data Drift:** Because e-commerce is highly dynamic (e.g. competitor bids affecting PPC efficiency), establish a monthly Kolmogorov-Smirnov (K-S) test on incoming distributions. The models should auto-retrain if the production RMSE degrades by >15%.

#### Outline of Project
- [Capstone_Sales_Prediction.ipynb](Capstone_Sales_Prediction.ipynb) (Primary Notebook: EDA, Regression, Classification, LSTM Forecasting, and Visual Metric Comparisons)
- [Extra_Multi-Class_Classification_with_Softmax.ipynb](Extra_Multi-Class_Classification_with_Softmax.ipynb) (Dedicated Deep Learning Multi-Class Softmax Categorization & Detailed Evaluation)

---

#### 🐳 Deployment

The trained models are served via a FastAPI REST API (`app.py`) and packaged in a Docker container.

> **Prerequisite:** Run the `8. Deployment` cell in `Capstone_Sales_Prediction.ipynb` **and** the final deployment cell in `Extra_Multi-Class_Classification_with_Softmax.ipynb` to generate all required model artifacts in the `/models` folder before building the Docker image.

**Build & Run:**
```bash
docker build -t capstone-sales-prediction .
docker run -p 8000:8000 capstone-sales-prediction
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

| Endpoint | Model | Metric | Description |
|---|---|---|---|
| `/predict/units_sold` | XGBoost Regressor | RMSE ~31.21 | Daily units sold forecast per product |
| `/predict/high_demand` | SVC Classifier | F1-Score ~0.80 | High demand spike classification |
| `/predict/store_volume` | Bidirectional LSTM | RMSE ~51.17 (aggregate) | Store-wide daily aggregate volume forecast |
| `/predict/product_category` | Softmax Neural Network | Confusion Matrix / Classification Report | Product category classification from financial heuristics |

##### Contact and Further Information
For any further inquiries, view the Jupyter notebooks or reach out via standard channels.
