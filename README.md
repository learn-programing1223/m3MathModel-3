# Memphis Neighborhood Vulnerability Analysis

## Project Overview
This project analyzes neighborhood vulnerability in Memphis using machine learning techniques to identify areas that may need additional support or resources. The analysis combines various socioeconomic and geographic factors to create a comprehensive vulnerability score for each neighborhood.

## Features
- Data preprocessing and cleaning of neighborhood statistics
- Advanced feature engineering to capture complex relationships
- Machine learning model optimization using Optuna
- Ensemble modeling with XGBoost, SVR, and Neural Networks
- Visualization of results and feature importance
- Ranked vulnerability scores with contributing factors for each neighborhood

## Files
- `main.py` - Primary script with data processing and modeling code
- `incomeSheet.csv` - Input data with neighborhood statistics
- `memphis_vulnerability_ranking.csv` - Output file with ranked neighborhoods and vulnerability scores
- `feature_importances.png` - Visualization of the most influential factors
- `actual_vs_predicted.png` - Model accuracy visualization
- Other visualization files for feature correlations and importance

## Key Features Analyzed
- Elderly population percentage
- Under-18 population percentage
- Household vehicle access
- Distance from city center
- Median household income
- Tree canopy coverage
- Various interaction terms between these factors

## Model Details
The project uses an ensemble approach comparing:
- Optimized XGBoost Regressor
- Support Vector Regression
- Multi-layer Perceptron Neural Network
- Stacked ensemble of the above models

Hyperparameter optimization is performed using Optuna to find the best model configuration.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- optuna

## Usage
1. Ensure you have the required libraries installed
2. Place your data in `incomeSheet.csv`
3. Run the script with:
```
python main.py
```
4. Examine the output files for results

## Output
- A ranked list of neighborhoods by vulnerability score
- Top contributing factors for each neighborhood
- Visualizations of model accuracy and feature importance

## Methodology
The vulnerability score is calculated using a combination of socioeconomic factors weighted by their importance. Advanced feature engineering creates interaction terms and non-linear transformations to capture complex relationships between factors.

## Future Improvements
- Additional data sources to enhance feature set
- Geospatial analysis and mapping
- Time-series analysis to track changes in vulnerability
- Interactive dashboard for exploring results
