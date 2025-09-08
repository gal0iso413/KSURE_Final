# KSURE Credit Risk Model Pipeline

## 📋 Project Overview

This is a comprehensive credit risk prediction model for KSURE (Korea Trade Insurance Corporation). The model predicts company default risk over 1-4 year horizons using financial data, trade statistics, credit ratings, and macroeconomic indicators.

## 🏗️ Project Structure

```
KSURE_Final/
├── 📁 src/                             # All Python source code
│   ├── 1_Dataset.py                    # ⭐ START HERE: Dataset creation
│   ├── 1_Split.py                      # Data splitting (train/validation/OOT)
│   │
│   ├── 2_Model1_baseline.py            # 🔍 COMPARISON MODELS (Optional)
│   ├── 2_Model1_evaluate.py            # Model evaluation
│   ├── 2_Model1_explore.py             # Model exploration
│   ├── 2_Model2.py                     # Feature reduction experiments
│   ├── 2_Model3.py                     # Temporal validation experiments
│   ├── 2_Model4.py                     # Model comparison experiments
│   ├── 2_Model5.py                     # Class imbalance experiments
│   ├── 2_Model6.py                     # Architecture experiments
│   ├── 2_Model7.py                     # Hyperparameter experiments
│   │
│   ├── 3_Model8.py                     # 🎯 MAIN PIPELINE: Advanced modeling
│   ├── 4_Model9.py                     # Final model training
│   ├── 5_Model10.py                    # Production model & grading
│   │
│   ├── external_data.py                # External data processing utilities
│   ├── 재무제표_process.py             # Financial statement processing
│   └── 조기경보_process.py             # Early warning processing
│
├── 📁 data/                            # All data files
│   ├── raw/                            # Original source data
│   │   ├── 청약.csv                    # Insurance contracts (base table)
│   │   ├── 조기경보이력_리스크단계.csv  # Risk outcomes (Y variables)
│   │   ├── KED가공재무DATA.csv         # Financial data
│   │   ├── KED종합신용정보.csv         # Credit ratings
│   │   ├── 무역통계진흥원수출입실적.csv # Trade statistics
│   │   ├── 업종코드_수출자.csv         # Industry codes
│   │   ├── gdp_data.csv                # GDP data
│   │   ├── trade_data.csv              # Trade data
│   │   └── exchange_rate_data.csv      # Exchange rate data
│   │
│   ├── processed/                      # Processed datasets
│   │   ├── credit_risk_dataset.csv     # Main dataset from 1_Dataset.py
│   │   └── credit_risk_dataset_selected.csv # Selected features
│   │
│   └── splits/                         # Data splits from 1_Split.py
│       ├── train_data.csv              # Training data
│       ├── validation_data.csv         # Validation data
│       ├── oot_data.csv                # Out-of-time test data
│       └── split_metadata.json         # Split information
│
├── 📁 models/                          # Trained models
│   ├── baseline/                       # Models from 2_Model* experiments
│   ├── intermediate/                   # Models from development steps
│   └── final/                          # Production-ready models
│
├── 📁 results/                         # All outputs and results
│   ├── step1_baseline/                 # Baseline model results
│   ├── step2_feature_reduction/        # Feature selection results
│   ├── step3_temporal_validation/      # Temporal validation results
│   ├── step4_model_comparison/         # Model comparison results
│   ├── step5_class_imbalance/          # Class imbalance handling results
│   ├── step6_model_architecture/       # Architecture comparison results
│   ├── step7_optuna/                   # Hyperparameter optimization results
│   ├── step8_post/                     # Advanced modeling results
│   ├── validation/                     # Model validation results
│   └── grading/                        # Final grading and predictions
│
├── 📁 sql/                             # SQL scripts for data extraction
│   ├── KED재무DATA.sql
│   ├── KED종합신용정보.sql
│   ├── 무역통계진흥원수출입실적_test.sql
│   ├── 조기경보내역.sql
│   └── 청약_test.sql
│
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🚀 How to Run the Pipeline

### **MAIN PIPELINE (Recommended)**

For production use, run the main pipeline in this order:

```bash
cd src/

# Step 1: Data Preparation
python 1_Dataset.py          # Create comprehensive dataset
python 1_Split.py             # Split data (train/validation/OOT)

# Step 2: Skip 2_Model* files (they are comparison experiments)

# Step 3: Main Model Pipeline
python 3_Model8.py            # Advanced modeling techniques
python 4_Model9.py            # Final model training
python 5_Model10.py           # Production model & grade assignment
```

### **EXPERIMENTAL PIPELINE (Optional)**

If you want to explore different modeling approaches:

```bash
cd src/

# After running 1_Dataset.py and 1_Split.py...

# Model Comparison Experiments (Optional)
python 2_Model1_baseline.py   # Baseline models (XGBoost, MLP, RandomForest)
python 2_Model1_evaluate.py   # Evaluate baseline models
python 2_Model1_explore.py    # Explore model performance
python 2_Model2.py            # Feature reduction experiments
python 2_Model3.py            # Temporal validation experiments
python 2_Model4.py            # Model comparison experiments
python 2_Model5.py            # Class imbalance handling experiments
python 2_Model6.py            # Architecture comparison experiments
python 2_Model7.py            # Hyperparameter optimization experiments

# Then continue with main pipeline (3_Model8.py → 4_Model9.py → 5_Model10.py)
```

## 📊 Key Features

### **Data Sources**
- **Insurance Contracts**: Base table with contract information
- **Risk Outcomes**: Historical default/risk events (1-4 year horizons)
- **Financial Data**: Balance sheet and income statement ratios
- **Credit Ratings**: External credit assessment data
- **Trade Statistics**: Import/export performance data
- **Macroeconomic Data**: GDP, exchange rates, trade volumes
- **Industry Classifications**: Sector-specific risk factors

### **Model Architecture**
- **Multi-horizon Prediction**: Separate risk predictions for 1, 2, 3, and 4 years
- **Temporal Validation**: Strict time-based data splitting to prevent leakage
- **Feature Engineering**: Lookback periods, change rates, and temporal aggregations
- **Class Imbalance Handling**: Advanced techniques for rare default events
- **Ensemble Methods**: Multiple algorithms with optimal weighting

### **Risk Grading System**
- **10-Grade Scale**: AAA (lowest risk) to D (highest risk)
- **Monotonic Grading**: Higher grades correspond to higher observed default rates
- **Business Rules**: Interpretable grade assignment logic
- **Validation Framework**: Out-of-time testing for unbiased performance assessment

## 🎯 Key Outputs

After running the pipeline, you will have:

1. **Trained Models**: Production-ready XGBoost models for each time horizon
2. **Risk Grades**: A-D grade assignments for all companies
3. **Predictions**: Risk scores and probabilities for each company
4. **Validation Reports**: Comprehensive model performance analysis
5. **Business Insights**: Interpretable risk factors and grade explanations

## ⚠️ Important Notes

### **Current Status**
- ✅ Files have been reorganized into logical directory structure
- ⚠️ **File paths in Python scripts need updating** (currently in progress)
- ✅ All original functionality preserved

### **Data Requirements**
- All source data files should be placed in `data/raw/` directory
- Ensure Korean character encoding is properly handled (UTF-8)
- Verify date formats are consistent across all data sources

### **Execution Environment**
- Python 3.8+ required
- Install dependencies: `pip install -r requirements.txt`
- Recommended: 16GB+ RAM for large dataset processing
- GPU optional but recommended for faster model training

## 🔧 Troubleshooting

### **Common Issues**
1. **File Path Errors**: Update file paths in scripts to reflect new directory structure
2. **Memory Issues**: Consider reducing dataset size or using batch processing
3. **Korean Font Issues**: Install appropriate Korean fonts for visualization
4. **Missing Data**: Verify all source files are present in `data/raw/`

### **Next Steps for Setup**
1. Update file paths in all Python scripts
2. Test data loading and processing
3. Verify model training pipeline
4. Validate final outputs and grades

## 📞 Support

This model pipeline represents a comprehensive credit risk assessment system designed for production use at KSURE. The modular design allows for easy maintenance, updates, and handover to new team members.

---

*Last Updated: 2024 - KSURE Credit Risk Modeling Team*
