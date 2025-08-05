# Credit Risk Dataset Creator

## 📋 Overview

This tool creates a time-series aware dataset for credit risk prediction by combining:
- **Insurance contracts** (base table) 
- **Future risk outcomes** (Y variables: Year1-4 risk predictions)
- **Historical predictors** (X variables: financial, macro, trade data)

### Key Features
- ✅ **No data leakage**: Temporal alignment ensures future data doesn't leak into predictions
- ✅ **Mixed frequency support**: Handles daily/monthly/quarterly/annual data naturally
- ✅ **Flexible column mapping**: Different tables can use different column names
- ✅ **Firm-specific vs Market-level**: Handles both types of data automatically
- ✅ **XGBoost ready**: Output format optimized for tree-based models
- ✅ **Ordinal risk levels**: 0 (no risk) → 1 → 2 → 3 (maximum risk)
- ✅ **Smart missing handling**: 0 vs NaN distinction for no-risk vs no-data

---

## 🛠️ Installation

```bash
pip install pandas numpy datetime
```

---

## 📁 Data Requirements

### 1. Base Table (Contracts)
**File**: `contracts.csv`
```csv
com_id,firm_id,contract_date,other_columns...
COM001,FIRM001,2023-01-15,value1
COM002,FIRM002,2023-02-20,value2
```

**Required data**: (column names configurable)
- **Risk identifier**: ID for joining with risk outcomes (any column name) 
- **Firm identifier**: ID for joining with historical predictors (any column name)
- **Contract date**: Contract start date in YYYY-MM-DD format (any column name)

**Note**: Base table supports dual ID structure - one ID for Y variables (risk outcomes) and another ID for X variables (predictors)

### 2. Risk Table (Future Outcomes)
**File**: `risk_outcomes.csv`
```csv
com_id,start_date,end_date,risk_level
COM001,2024-03-10,2024-06-15,2
COM001,2024-08-15,2024-12-20,3
COM002,2024-01-05,2024-04-10,1
```

**Required data**: (column names configurable)
- **Risk identifier**: Must match risk IDs in base table (can be different column name)
- **Risk start date**: When risk period began in YYYY-MM-DD format (any column name)
- **Risk end date**: When risk period ended in YYYY-MM-DD format (any column name)  
- **Risk level**: Ordinal risk level (1, 2, 3 - higher = worse) (any column name)

**Note**: Risk table uses date ranges (start_date to end_date) rather than single event dates

### 3. X Variable Tables (Historical Predictors)

#### Firm-Specific Data (e.g., Financial)
**File**: `financial_data.csv`
```csv
firm_code,report_date,revenue,profit,debt_ratio
FIRM001,2022-12-31,1000000,50000,0.3
FIRM001,2023-03-31,1100000,55000,0.32
```

**Required data**: (column names configurable)
- **Firm identifier**: Must match firms in base table (can be different column name)
- **Date**: When data was recorded (any column name)
- **Features**: Any number of numeric/categorical features

#### Market-Level Data (e.g., Macro Economic)
**File**: `macro_indicators.csv`
```csv
date,gdp_growth,inflation_rate,interest_rate
2023-01-01,2.5,3.2,1.75
2023-02-01,2.3,3.1,1.80
```

**Required data**: (column names configurable)
- **Date**: When data was recorded (any column name)
- **Features**: Any number of economic indicators
- **No firm identifier**: Same values apply to all firms on same date

#### Market-Level Data (e.g., Trade)
**File**: `trade_data.csv`
```csv
trade_date,export_growth,import_growth,trade_balance
2023-01-01,5.2,3.1,2500000
2023-02-01,4.8,3.5,2300000
```

**Required data**: (column names configurable)
- **Date**: Trade date (can be any column name)
- **Features**: Trade-related metrics
- **No firm identifier**: Market-level data shared across all firms

---

## 🔧 Flexible Column Mapping

The tool supports **different column names across tables**, making it easy to work with real-world data sources:

### Example Scenario:
```python
# Your actual data might have different naming conventions:
contracts.csv:     com_id, firm_id, contract_date
risk_data.csv:     company_id, start_date, end_date, severity  
financial.csv:     firm_code, report_date, revenue
macro_data.csv:    date, gdp_growth (no firm column - market level)
trade_data.csv:    trade_date, exports (no firm column - market level)
```

### Solution:
```python
'column_mappings': {
    'base_table': {
        'date_column': 'contract_date', 
        'risk_id_column': 'com_id',      # For risk table joins
        'firm_id_column': 'firm_id'      # For X variable joins
    },
    'risk_table': {
        'start_date_column': 'start_date', 
        'end_date_column': 'end_date',
        'risk_id_column': 'company_id', 
        'risk_level_column': 'severity'
    },
    'x_tables': {
        'financial': {'date_column': 'report_date', 'firm_id_column': 'firm_code'},
        'macro': {'date_column': 'date', 'firm_id_column': None},  # Market-level
        'trade': {'date_column': 'trade_date', 'firm_id_column': None}  # Market-level
    }
}
```

**Benefits:**
- ✅ No need to rename columns in your source data
- ✅ Handles firm-specific AND market-level data  
- ✅ Clear distinction between data types
- ✅ Easy to add new data sources

---

## ⚙️ Configuration

Create your configuration dictionary:

```python
config = {
    # === FILE PATHS ===
    'base_table_path': 'data/contracts.csv',
    'risk_table_path': 'data/risk_outcomes.csv',
    'x_variable_paths': {
        'financial': 'data/financial_data.csv',
        'macro': 'data/macro_indicators.csv',
        'trade': 'data/trade_data.csv'
    },
    
    # === TABLE-SPECIFIC COLUMN MAPPINGS ===
    'column_mappings': {
        'base_table': {
            'date_column': 'contract_date',
            'risk_id_column': 'com_id',            # ID for joining Y variables (risk outcomes)
            'firm_id_column': 'firm_id'            # ID for joining X variables (predictors)
        },
        'risk_table': {
            'start_date_column': 'start_date',
            'end_date_column': 'end_date',
            'risk_id_column': 'com_id',            # Should match risk_id_column from base_table
            'risk_level_column': 'risk_level'
        },
        'x_tables': {
            'financial': {
                'date_column': 'report_date',
                'firm_id_column': 'firm_code'      # Different naming convention
            },
            'macro': {
                'date_column': 'date',
                'firm_id_column': None             # Market-level data, no firm ID
            },
            'trade': {
                'date_column': 'trade_date',       # Different date column name
                'firm_id_column': None             # Market-level data, no firm ID
            }
        }
    },
    
    # === FEATURE ENGINEERING ===
    # How many historical periods to include for each data type
    'lookback_periods': {
        'financial': 8,    # 8 quarters (2 years of quarterly data)
        'macro': 12,       # 12 months (1 year of monthly data)
        'trade': 6         # 6 months (recent trade trends)
    },
    
    # Which future years to predict (1=next year, 2=year after, etc.)
    'prediction_horizons': [1, 2, 3, 4],
    
    # === COLUMN SELECTION (OPTIONAL) ===
    # Specify exactly which columns to include (if not specified, uses all columns)
    'x_include_columns': {
        'financial': ['revenue', 'profit', 'debt_ratio', 'cash_flow'],  # Only these columns
        'macro': ['gdp_growth', 'inflation_rate', 'interest_rate'],     # Only these columns
        'trade': None  # None = include all columns (default behavior)
    },
    
    # Additional columns to exclude (applied after include filter)
    'x_exclude_columns': {
        'financial': ['notes', 'updated_by'],  # Exclude non-predictive columns
        'macro': [],                           # No additional exclusions
        'trade': ['source']                    # Exclude metadata columns
    }
}
```

---

## 🎯 Column Selection Guide

You can now precisely control which columns to use from each data source:

### **Option 1: Include All Columns (Default)**
```python
'x_include_columns': {
    'financial': None,  # Uses all columns from financial table
    'macro': None       # Uses all columns from macro table
}
```

### **Option 2: Select Specific Columns**
```python
'x_include_columns': {
    'financial': ['revenue', 'profit', 'debt_ratio'],  # Only these 3 columns
    'macro': ['gdp_growth', 'inflation_rate']          # Only these 2 columns
}
```

### **Option 3: Combine Include + Exclude**
```python
'x_include_columns': {
    'financial': ['revenue', 'profit', 'debt_ratio', 'cash_flow', 'notes']  # 5 columns
},
'x_exclude_columns': {
    'financial': ['notes']  # Remove notes → final result: 4 columns
}
```

### **Processing Order:**
1. **Start** with all columns in the table
2. **Apply include filter** (if specified) → only keep listed columns
3. **Apply exclude filter** → remove unwanted columns
4. **Remove system columns** → date, firm_id are never included as features

### **Example Output:**
```python
# If financial.csv has: [report_date, firm_code, revenue, profit, debt_ratio, notes, updated_by]
# And config is:
'x_include_columns': {'financial': ['revenue', 'profit', 'debt_ratio']}

# Final features will be:
financial_revenue_t0, financial_revenue_t1, financial_revenue_t2, ...
financial_profit_t0, financial_profit_t1, financial_profit_t2, ...
financial_debt_ratio_t0, financial_debt_ratio_t1, financial_debt_ratio_t2, ...
```

**Benefits:**
- ✅ **Precise control** over feature selection
- ✅ **Avoid noisy features** by selecting only relevant columns
- ✅ **Domain knowledge** can guide feature selection
- ✅ **Backward compatible** - if not specified, uses all columns

**Logging Output:**
```
📊 Processing financial data...
      🎯 Include filter: 3/3 columns found
      🚫 Exclude filter: removed 1 columns
      ✅ Added 24 features from financial (3 columns × 8 lookback periods)
```

---

## 🚀 Execution Guide

### Step 1: Import and Initialize
```python
from Dataset import DatasetCreator

# Create your config (see Configuration section above)
config = {...}

# Initialize creator
creator = DatasetCreator(config)
```

### Step 2: Create Dataset
```python
# Option A: Create everything in one step
dataset = creator.create_dataset()

# Option B: Step-by-step (for debugging)
creator.load_base_table()
creator.join_y_variables()
creator.join_x_variables()
dataset = creator.final_dataset
```

### Step 3: Save Results
```python
# Save to CSV
creator.save_dataset('output/credit_risk_dataset.csv')

# Or save manually
dataset.to_csv('my_dataset.csv', index=False)
```

### Complete Example
```python
from Dataset import DatasetCreator

# Configuration with flexible column mappings
config = {
    'base_table_path': 'data/contracts.csv',
    'risk_table_path': 'data/risk_outcomes.csv',
    'x_variable_paths': {
        'financial': 'data/financial_data.csv',
        'macro': 'data/macro_indicators.csv'
    },
    'column_mappings': {
        'base_table': {
            'date_column': 'contract_date',
            'risk_id_column': 'com_id',        # For risk table joins
            'firm_id_column': 'firm_id'        # For X variable joins
        },
        'risk_table': {
            'start_date_column': 'start_date',
            'end_date_column': 'end_date',
            'risk_id_column': 'com_id',        # Matches base table risk_id_column
            'risk_level_column': 'risk_level'
        },
        'x_tables': {
            'financial': {
                'date_column': 'report_date',
                'firm_id_column': 'firm_code'   # Different column name!
            },
            'macro': {
                'date_column': 'date',
                'firm_id_column': None          # Market-level data
            }
        }
    },
    'lookback_periods': {
        'financial': 8,
        'macro': 12
    },
    'prediction_horizons': [1, 2, 3, 4],
    'x_include_columns': {
        'financial': ['revenue', 'profit', 'debt_ratio'],  # Select specific columns
        'macro': None  # Use all macro columns
    }
}

# Create dataset
creator = DatasetCreator(config)
dataset = creator.create_dataset()
creator.save_dataset('output/final_dataset.csv')

print("✅ Dataset ready for XGBoost!")
```

---

## 📊 Output Format

### Y Variables (Target)
- `risk_year1`: Maximum risk in first 1 year after contract (0,1,2,3, or NaN)
- `risk_year2`: Maximum risk in first 2 years after contract (cumulative)
- `risk_year3`: Maximum risk in first 3 years after contract (cumulative)
- `risk_year4`: Maximum risk in first 4 years after contract (cumulative)

**Risk Assignment Logic:**
- **0**: Complete prediction period with no risk events observed
- **1,2,3**: Complete prediction period with risk events (max level observed)
- **NaN**: Prediction period not completed yet (insufficient time passed)

*All risk levels require complete prediction periods for fair comparison.*

### X Variables (Features)
Feature naming: `{data_type}_{variable}_{time_period}`

Examples:
- `financial_revenue_t0`: Most recent revenue (t0 = current)
- `financial_revenue_t1`: Revenue 1 period ago
- `macro_gdp_growth_t0`: Most recent GDP growth
- `trade_export_growth_t2`: Export growth 2 periods ago

---

## 🔍 Understanding the Logic

### Y Variable Creation (Future Risk)
```python
# For each contract on 2023-01-15 (cumulative prediction windows):
# Year1 window: 2023-01-15 to 2024-01-15 (1 year total)
# Year2 window: 2023-01-15 to 2025-01-15 (2 years total)  
# Year3 window: 2023-01-15 to 2026-01-15 (3 years total)
# Year4 window: 2023-01-15 to 2027-01-15 (4 years total)

# For each window, find risk periods that overlap:
# Risk overlap: risk_start_date < window_end AND risk_end_date > window_start
# If multiple overlapping risks: take maximum risk level
# If no overlapping risks but data coverage exists: 0 (no risk)
# If insufficient data coverage: NaN (unknown)
```

### X Variable Creation (Historical Predictors)
```python
# For contract on 2023-06-15 with lookback_periods={'financial': 4}:
# t0: Most recent financial data before 2023-06-15
# t1: Most recent financial data before 2023-05-15 (1 month back)
# t2: Most recent financial data before 2023-04-15 (2 months back)  
# t3: Most recent financial data before 2023-03-15 (3 months back)

# This creates 4 time points of historical data per variable
```

---

## ⚠️ Troubleshooting

### Common Issues

**1. "KeyError: column not found"**
```python
# Check your column names match the config
print(df.columns.tolist())  # See actual column names
```

**2. "No data available"**
```python
# Check date ranges
print(f"Contracts: {contracts.date_col.min()} to {contracts.date_col.max()}")
print(f"Risk data: {risk_data.date_col.min()} to {risk_data.date_col.max()}")
```

**3. "Too many NaN values"**
```python
# Adjust lookback periods or date ranges
config['lookback_periods'] = {'financial': 4}  # Reduce from 8 to 4
```

**4. "DatetimeIndex error"** 
```python
# Ensure date columns are in YYYY-MM-DD format
df['date_col'] = pd.to_datetime(df['date_col'])
```

### Debug Mode
```python
# Add debug prints
creator = DatasetCreator(config)
creator.load_base_table()
print(f"Base data shape: {creator.base_data.shape}")

creator.join_y_variables() 
print(f"After Y join: {creator.base_data.shape}")

creator.join_x_variables()
print(f"Final shape: {creator.final_dataset.shape}")
```

---

## 📈 Next Steps

After creating your dataset:

1. **Data Quality Check**:
   ```python
   # Check missing data
   missing_pct = dataset.isna().sum() / len(dataset) * 100
   print(missing_pct.sort_values(ascending=False).head(10))
   ```

2. **XGBoost Training**:
   ```python
   import xgboost as xgb
   
   # Prepare features and targets
   feature_cols = [col for col in dataset.columns if not col.startswith('risk_year')]
   X = dataset[feature_cols]
   y = dataset['risk_year1']
   
   # Train model
   model = xgb.XGBClassifier()
   model.fit(X, y)
   ```

3. **Feature Importance Analysis**:
   ```python
   # See which features matter most
   importance = model.feature_importances_
   feature_importance = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
   print(feature_importance.head(20))
   ```

---

## 🎯 Tips for Success

1. **Start Small**: Begin with 1-2 data sources and 1 prediction year
2. **Check Data Quality**: Ensure reasonable date ranges and no obvious errors
3. **Experiment with Lookback**: Different industries may need different historical periods
4. **Monitor Missing Data**: >50% missing usually indicates data alignment issues
5. **Validate Logic**: Manually check a few examples to ensure correctness

---

## 📞 Support

If you encounter issues:
1. Check the configuration matches your data structure
2. Verify date formats (YYYY-MM-DD)
3. Ensure risk_ids match between base table and risk table
4. Ensure firm_ids match between base table and X variable tables  
5. Review the troubleshooting section
6. Use debug mode to isolate problems

**Happy modeling! 🚀**