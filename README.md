# Employee Attrition Prediction Model

## Overview
This project implements a machine learning solution to predict employee attrition using the IBM HR Analytics dataset. The model identifies employees at risk of leaving the organization based on various demographic, job-related, and satisfaction metrics.

## Dataset
The project uses the IBM HR Analytics Employee Attrition dataset, containing 1,470 employee records with 35 features including:
- Demographics (age, gender, marital status)
- Job-related metrics (department, role, years at company)
- Satisfaction scores
- Work environment factors (overtime, travel)
- Compensation details

## Features
The model leverages both original features and engineered features:

### Key Original Features
- Age
- MonthlyIncome
- YearsAtCompany
- DistanceFromHome
- JobSatisfaction
- WorkLifeBalance
- Department
- JobRole
- OverTime
- MaritalStatus

### Engineered Features
- IncomeToJobLevelRatio
- CareerProgressionRatio 
- SatisfactionComposite
- CompanyLoyaltyRatio
- CareerStagnationRatio
- PromotionRate
- IncomeToDistanceRatio

## Models
The project implements and compares several machine learning models:
- Logistic Regression (AUC: 0.816)
- Support Vector Machine - Linear (AUC: 0.818)
- Linear Discriminant Analysis (AUC: 0.806)
- Random Forest (AUC: 0.748)
- XGBoost (AUC: 0.759)
- Quadratic Discriminant Analysis (AUC: 0.580)
- Cubic Classifier (AUC: 0.679)

## Key Findings
- **Work-life factors**: Overtime, business travel, and commute distance are top predictors
- **Demographics**: Younger and single employees have higher attrition rates
- **Job roles**: Sales Representatives (39.8%) and Laboratory Technicians (23.9%) have highest attrition
- **Linear relationships**: Linear models outperform complex models, suggesting linear relationships between features and attrition
- **Satisfaction metrics**: Job satisfaction and work-life balance strongly influence retention

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

The key dependencies include:
- numpy==1.23.5
- pandas==1.5.3
- scikit-learn==1.2.2
- matplotlib==3.7.1
- seaborn==0.12.2
- xgboost==1.7.5
- shap==0.41.0
- imbalanced-learn==0.10.1

### Running the Notebook
```bash
jupyter notebook notebooks/attrition_prediction_model.ipynb
```

## Usage
To use the prediction function on new employee data:

```python
from src.prediction import predict_attrition

# Example employee data (DataFrame with required features)
employee_data = pd.DataFrame({
    'Age': [35], 
    'MonthlyIncome': [5000],
    'OverTime': ['Yes'],
    # Include all other required features
})

# Get prediction
result = predict_attrition(employee_data)

print(f"Attrition Prediction: {'Yes' if result['prediction'][0] == 1 else 'No'}")
print(f"Probability: {result['probability'][0]:.2f}")
print(f"Risk Level: {result['risk_level'][0]}")
```

## Model Evaluation
The final model uses a custom threshold (0.35) to optimize the F1 score rather than the default 0.5 probability threshold. This provides better detection of potential attrition cases.

## Future Improvements
- Integrate model into an HR dashboard system
- Develop more sophisticated features using time-series employee data
- Create department-specific models for more targeted predictions
- Add explainability features for HR interpretation

## License
This project is proprietary and confidential to TAO-IQ.

## Acknowledgments
- The IBM HR Analytics dataset was used for model development
- This model was developed as part of a TAO-IQ partner evaluation project
