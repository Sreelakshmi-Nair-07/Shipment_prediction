# Shipment Booking Prediction

## Overview
This project analyzes shipment booking data from 2021-2025 and builds a multi-label classification model to predict shipment types for the next booking date. The analysis reveals that bookings occur daily for all companies, making date prediction deterministic. The focus shifts to predicting the mix of shipment types (Air, Express, International, Surface) using machine learning.

## Problem Statement
Predict the next booking date and shipment type for each company based on historical data. Since daily bookings are consistent, the model predicts shipment probabilities for the next day.

## Features
- **Time Series Analysis**: Visualizes daily booking trends and confirms 100% daily coverage per company.
- **Feature Engineering**: Extracts temporal features (weekdays, months, rolling statistics) to capture patterns.
- **Multi-Label Classification**: Handles multiple shipment types occurring on the same day.
- **Model Comparison**: Implements Random Forest (with hyperparameter tuning) and Logistic Regression.
- **Evaluation**: Uses precision-recall curves and confusion matrices, suitable for imbalanced classes.

## Installation and Usage
1. **Prerequisites**: Python 3.8+, Git.
2. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sreelakshmi-Nair-07/Shipment_prediction.git
   cd Shipment_prediction
   ```
3. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn pyyaml
   ```
4. **Run the Notebook**:
   - Open `shipement_prediction.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
   - Execute cells sequentially to reproduce the analysis and predictions.

## Data
- **Source**: `shipment_booking_data_2021_2025.csv` (historical booking records).
- **Structure**: Columns include company_name, booking_date, shipment_type.
- **Preprocessing**: Aggregated to daily level, engineered features for modeling.

## Models
- **Random Forest**: Tuned with GridSearchCV for optimal hyperparameters.
- **Logistic Regression**: With standard scaling.
- **Evaluation Metrics**: F1-score, precision, recall; handles class imbalance with balanced weights.

## Results
- Models perform well on dominant classes (Air, Surface) and reasonably on minorities (Express, International).
- Random Forest captures complex interactions; Logistic Regression offers interpretability.
- Predictions provide probabilities and threshold-based shipment recommendations.

## Key Insights
- Bookings are daily without gaps, so next date = last date + 1 day.
- Shipment patterns vary by company, day of week, and season.
- Multi-label approach reflects real logistics where shipments coexist.

## Conclusion
This project demonstrates end-to-end ML for logistics prediction, from EDA to deployment-ready predictions. It highlights the importance of temporal features and appropriate evaluation for imbalanced data.

## License
This project is open-source. Feel free to contribute or use for learning.

## Contact
For questions, reach out via GitHub issues.