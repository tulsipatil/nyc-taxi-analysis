# NYC Taxi Trip Analysis

Exploratory analysis and predictive modeling on **3M+ NYC yellow taxi trips** to uncover fare, tipping, and payment behavior patterns.

---

## What I Found

| Question | Finding |
|----------|---------|
| What drives high fares? | Trip distance, duration, and location predict fare level with **AUC = 0.99** |
| What drives tips? | Fare amount and distance are top predictors - but tipping is noisy (R² = 0.46), suggesting customer satisfaction matters more than trip features |
| How do people pay? | Credit card dominates (70%+) at all hours; cash rises slightly after midnight - payment mode predictable at **86.5% accuracy** |
| When are trips busiest? | Peak volume at 4-7 PM (rush hour); most trips are 0–3 miles - NYC taxi is a short-trip service |

---

## Dataset

- **Source:** NYC Taxi & Limousine Commission (TLC) - [tlc-trip-record-data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **File:** yellow_tripdata_2025-09.parquet (September 2025)
- **Size:** 3M+ trip records
- **Key fields:** pickup/dropoff timestamps, location IDs, trip distance, passenger count, fare amount, tip amount, payment type

---

## Analysis

### Step 1 — Data Cleaning
- Removed cancelled rides, logging errors, and extreme outliers (distance > 100 miles, duration > 4 hours)
- Enforced positive passenger count and total amount
- Imputed missing surcharge fields with 0 - interpreted as "no surcharge applied"
- **Result:** clean, reliable dataset ready for analysis

### Step 2 — Exploratory Data Analysis

**Trip distance distribution**
- Distribution is highly right-skewed - 80%+ of trips fall within 0–3 miles
- Long tail (20+ miles) represents airport trips and suburban commutes
- Confirmed need for median-based thresholds rather than mean for fare labeling

**Fare behavior**
- High-fare trips have significantly longer distances (avg 5.4 mi) vs low-fare (avg 1.1 mi)
- High-fare trips also show higher average tip amounts and more surcharges (airport, congestion)

**Tip behavior**
- Tips increase with distance up to ~15 miles, then become inconsistent
- Fare amount is the single strongest predictor of tip amount
- ~46% of tip variance explained by observable trip features - remainder driven by unobserved factors

**Payment patterns**
- Credit card (type 1): dominant at all hours, peaking during afternoon rush
- Cash (type 2): slightly elevated 12am–4am, likely nightlife traffic
- Rare types (dispute, no-charge): consistently low, no hourly pattern

### Step 3 — Predictive Modeling

| Task | Approach | Result |
|------|----------|--------|
| Fare classification (high vs low) | Logistic Regression + hyperparameter tuning | AUC = 0.99 |
| Tip amount prediction | Random Forest Regressor | MAE = $1.92, R² = 0.46 |
| Payment type prediction | Random Forest Classifier | Accuracy = 86.5%, F1 = 0.803 |

---

## Key Business Insights

- **For drivers:** Long-distance trips (airport, suburbs) reliably produce high fares — predictable with 99% AUC
- **For operations:** Rush hour (4–7 PM) is peak demand - staffing and pricing decisions should prioritize this window
- **For payment strategy:** Card payments are dominant but cash increases after midnight — both channels need to remain available
- **On tipping:** Distance and fare alone don't explain tips well (R² = 0.46) - service quality is likely the missing factor

---

## Tech Stack

`Python` `PySpark` `Spark MLlib` `Pandas` `Matplotlib` `Seaborn` `Google Colab`

---

## Setup

```bash
git clone https://github.com/tulsipatil/nyc-taxi-analysis
cd nyc-taxi-analysis
pip install pyspark pandas matplotlib seaborn
```

Download data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page  
Update `path` in the notebook to your local parquet file location.
