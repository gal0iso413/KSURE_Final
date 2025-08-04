"""
XGBoost Risk Prediction Model - Step 3: Comprehensive Exploratory Data Analysis (EDA)
==================================================================================

Step 3 Implementation:
- Data Quality Analysis: Missing patterns, outliers, data types
- Temporal Patterns: Distribution changes over time, seasonality
- Target Variable Analysis: Class distribution over time, business cycles impact
- Feature Relationships: Correlation analysis, feature distributions by target class
- Business Logic Validation: Do the data patterns make business sense?
- Data Leakage Investigation: Identify potentially problematic features
- Goal: Deep understanding of data characteristics to inform all subsequent decisions

Design Focus:
- Comprehensive analysis without including previous/future steps
- Business-focused insights and interpretations
- Korean font support for visualizations
- Systematic approach to data understanding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
import warnings
import platform
import matplotlib.font_manager as fm
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Configure Korean font for matplotlib
def setup_korean_font():
    """Set up Korean font for matplotlib visualizations"""
    system = platform.system()
    
    if system == "Windows":
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
    elif system == "Darwin":  # macOS
        korean_fonts = ['AppleGothic', 'NanumGothic']
    else:  # Linux
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    korean_font = None
    
    for font in korean_fonts:
        if font in available_fonts:
            korean_font = font
            break
    
    if korean_font:
        plt.rcParams['font.family'] = korean_font
        print(f"✅ Korean font set: {korean_font}")
    else:
        print("⚠️  Preferred Korean fonts not found. Using fallback options...")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    return korean_font

# Set up Korean font
setup_korean_font()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON, handling Korean text and pandas objects
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


class ComprehensiveEDA:
    """
    Step 3: Comprehensive Exploratory Data Analysis for Risk Prediction
    
    Provides deep understanding of data characteristics through:
    - Data quality analysis
    - Temporal pattern analysis
    - Target variable analysis
    - Feature relationship analysis
    - Business logic validation
    - Data leakage investigation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize EDA framework
        
        Args:
            config: Dictionary containing EDA configuration
        """
        self.config = config
        self.data = None
        self.feature_columns = None
        self.target_columns = None
        self.temporal_features = None
        self.static_features = None
        self.korean_font = setup_korean_font()
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Risk level definitions
        self.risk_levels = {
            0: "위험없음 (No Risk)",
            1: "낮은위험 (Low Risk)", 
            2: "중간위험 (Medium Risk)",
            3: "높은위험 (High Risk)"
        }
        
        print("🔍 Comprehensive EDA Framework Initialized")
        print(f"📊 Target variables: {config['target_columns']}")
        print(f"🎯 Focus: Deep data understanding for informed modeling")
    
    def _create_results_directory(self) -> str:
        """Create results directory for saving outputs"""
        results_dir = "result/step3_eda"
        
        os.makedirs(results_dir, exist_ok=True)
        
        subdirs = ['data_quality', 'temporal_analysis', 'target_analysis', 
                  'feature_analysis', 'business_insights', 'visualizations']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir
    
    def _save_plot(self, filename: str, subdir: str = 'visualizations', dpi: int = 300) -> str:
        """Save current plot to results directory"""
        save_path = os.path.join(self.results_dir, subdir, f"{filename}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  💾 Saved: {filename}.png")
        return save_path
    
    def load_csv_with_korean_encoding(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with proper Korean encoding handling"""
        encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully loaded {file_path} with {encoding} encoding")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            print(f"⚠️  Loaded {file_path} with error handling")
            return df
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def load_and_prepare_data(self):
        """Load and prepare data for comprehensive EDA"""
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING & PREPARATION")
        print("="*60)
        
        # Load data
        self.data = self.load_csv_with_korean_encoding(self.config['data_path'])
        print(f"✅ Data loaded: {self.data.shape}")
        
        # Apply exclusions
        exclude_cols = self.config.get('exclude_columns', [])
        if exclude_cols:
            self.data = self.data.drop(columns=[col for col in exclude_cols if col in self.data.columns])
        
        # Identify column types
        self.target_columns = [col for col in self.data.columns if col.startswith('risk_year')]
        self.feature_columns = [col for col in self.data.columns if not col.startswith('risk_year')]
        
        # Separate temporal and static features
        self.temporal_features = [col for col in self.feature_columns if any(period in col for period in ['_t0', '_t1', '_t2'])]
        self.static_features = [col for col in self.feature_columns if col not in self.temporal_features]
        
        print(f"📊 Dataset structure:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Target variables: {len(self.target_columns)}")
        print(f"   • Feature columns: {len(self.feature_columns)}")
        print(f"   • Temporal features: {len(self.temporal_features)}")
        print(f"   • Static features: {len(self.static_features)}")
        
        return self.data
    
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        print("\n" + "="*60)
        print("2️⃣ DATA QUALITY ANALYSIS")
        print("="*60)
        
        # Missing value analysis
        print("\n🔍 MISSING VALUE ANALYSIS")
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        }).sort_values('Missing_Percent', ascending=False)
        
        # Save missing value analysis
        missing_df.to_csv(os.path.join(self.results_dir, 'data_quality', 'missing_value_analysis.csv'), 
                         index=False, encoding='utf-8-sig')
        
        print(f"   • Columns with missing data: {(missing_data > 0).sum()}")
        print(f"   • Total missing values: {missing_data.sum():,}")
        print(f"   • Average missing rate: {missing_percent.mean():.2f}%")
        
        # Create missing value heatmap
        plt.figure(figsize=(12, 8))
        missing_matrix = self.data.isnull()
        sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Value Pattern Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Records', fontsize=12)
        self._save_plot("01_missing_value_patterns", 'data_quality')
        
        # Outlier analysis for numeric features
        print("\n🔍 OUTLIER ANALYSIS")
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number]).columns
        
        outlier_summary = []
        for feature in numeric_features[:10]:  # Analyze first 10 features
            values = self.data[feature].dropna()
            if len(values) > 0:
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outlier_percent = (len(outliers) / len(values)) * 100
                
                outlier_summary.append({
                    'Feature': feature,
                    'Outlier_Count': len(outliers),
                    'Outlier_Percent': outlier_percent,
                    'Min_Value': values.min(),
                    'Max_Value': values.max(),
                    'Mean_Value': values.mean()
                })
        
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df.to_csv(os.path.join(self.results_dir, 'data_quality', 'outlier_analysis.csv'), 
                         index=False, encoding='utf-8-sig')
        
        print(f"   • Features analyzed for outliers: {len(outlier_summary)}")
        print(f"   • Average outlier rate: {outlier_df['Outlier_Percent'].mean():.2f}%")
        
        # Data type analysis
        print("\n🔍 DATA TYPE ANALYSIS")
        dtype_counts = self.data.dtypes.value_counts()
        print(f"   • Data types: {safe_json_serialize(dict(dtype_counts))}")
        
        # Save data quality summary
        quality_summary = {
            'total_records': len(self.data),
            'total_features': len(self.feature_columns),
            'missing_columns': (missing_data > 0).sum(),
            'total_missing_values': missing_data.sum(),
            'average_missing_rate': missing_percent.mean(),
            'average_outlier_rate': outlier_df['Outlier_Percent'].mean() if len(outlier_df) > 0 else 0,
            'data_types': {str(k): int(v) for k, v in dtype_counts.items()}
        }
        
        import json
        with open(os.path.join(self.results_dir, 'data_quality', 'quality_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(quality_summary), f, indent=2, ensure_ascii=False)
        
        return quality_summary
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        print("\n" + "="*60)
        print("3️⃣ TEMPORAL PATTERN ANALYSIS")
        print("="*60)
        
        # Extract time periods from feature names
        time_periods = ['t0', 't1', 't2']
        feature_groups = {}
        
        for period in time_periods:
            period_features = [col for col in self.temporal_features if f'_{period}' in col]
            if period_features:
                feature_groups[period] = period_features
        
        print(f"📊 Temporal feature groups: {list(feature_groups.keys())}")
        
        # Create temporal evolution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Feature Evolution Analysis', fontsize=16, fontweight='bold')
        
        # Sample a few key features for visualization
        sample_features = []
        for feature in self.temporal_features[:]:
            base_name = feature.split('_t')[0]
            if base_name not in [f.split('_t')[0] for f in sample_features]:
                sample_features.append(feature)
        
        for i, feature in enumerate(sample_features[:4]): # First 4 features
            ax = axes[i//2, i%2]
            base_name = feature.split('_t')[0]
            
            values_by_period = []
            periods = []
            
            for period in time_periods:
                period_feature = f"{base_name}_{period}"
                if period_feature in self.data.columns:
                    values = self.data[period_feature].dropna()
                    if len(values) > 0:
                        values_by_period.append(values)
                        periods.append(period)
            
            if len(values_by_period) > 1:
                ax.boxplot(values_by_period, labels=periods)
                ax.set_title(f'{base_name} - Temporal Evolution', fontsize=12)
                ax.set_ylabel('Value', fontsize=10)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("02_temporal_evolution", 'temporal_analysis')
        
        # Correlation analysis across time periods
        print("\n📈 TEMPORAL CORRELATION ANALYSIS")
        temporal_correlations = {}
        
        for base_name in set([f.split('_t')[0] for f in self.temporal_features]):
            period_values = {}
            
            for period in time_periods:
                period_feature = f"{base_name}_{period}"
                if period_feature in self.data.columns:
                    period_values[period] = self.data[period_feature]
            
            if len(period_values) > 1:
                # Calculate correlations between periods
                period_df = pd.DataFrame(period_values)
                corr_matrix = period_df.corr()
                temporal_correlations[base_name] = corr_matrix
        
        # Save temporal correlations
        import json
        with open(os.path.join(self.results_dir, 'temporal_analysis', 'temporal_correlations.json'), 'w', encoding='utf-8') as f:
            json_data = {}
            for feature, corr_matrix in temporal_correlations.items():
                json_data[feature] = corr_matrix.to_dict()
            json.dump(safe_json_serialize(json_data), f, indent=2, ensure_ascii=False)
        
        print(f"   • Features with temporal correlation: {len(temporal_correlations)}")
        
        return temporal_correlations
    
    def analyze_target_variables(self):
        """Comprehensive target variable analysis"""
        print("\n" + "="*60)
        print("4️⃣ TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target_analysis = {}
        
        for target in self.target_columns:
            print(f"\n🎯 Analyzing {target}...")
            
            target_data = self.data[target].dropna()
            target_analysis[target] = {
                'sample_size': len(target_data),
                'distribution': target_data.value_counts().sort_index().to_dict(),
                'distribution_percent': (target_data.value_counts().sort_index() / len(target_data) * 100).to_dict(),
                'unique_values': sorted(target_data.unique()),
                'missing_count': self.data[target].isnull().sum(),
                'missing_percent': (self.data[target].isnull().sum() / len(self.data)) * 100
            }
            
            print(f"   • Sample size: {target_analysis[target]['sample_size']:,}")
            print(f"   • Distribution: {target_analysis[target]['distribution']}")
            print(f"   • Missing rate: {target_analysis[target]['missing_percent']:.2f}%")
        
        # Create target distribution visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Variable Distribution Analysis', fontsize=16, fontweight='bold')
        
        for i, target in enumerate(self.target_columns):
            ax = axes[i//2, i%2]
            
            target_data = self.data[target].dropna()
            value_counts = target_data.value_counts().sort_index()
            
            bars = ax.bar(value_counts.index, value_counts.values, alpha=0.7)
            ax.set_title(f'{target} - Risk Level Distribution', fontsize=12)
            ax.set_xlabel('Risk Level', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels
            total = len(target_data)
            for bar, (level, count) in zip(bars, value_counts.items()):
                percentage = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*total,
                       f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self._save_plot("03_target_distributions", 'target_analysis')
        
        # Save target analysis
        import json
        with open(os.path.join(self.results_dir, 'target_analysis', 'target_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(target_analysis), f, indent=2, ensure_ascii=False)
        
        return target_analysis
    
    def analyze_feature_relationships(self):
        """Analyze feature relationships and correlations"""
        print("\n" + "="*60)
        print("5️⃣ FEATURE RELATIONSHIP ANALYSIS")
        print("="*60)
        
        # Correlation analysis
        print("\n📊 CORRELATION ANALYSIS")
        
        # Use only numeric features
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()
        
        # Save correlation matrix
        correlation_matrix.to_csv(os.path.join(self.results_dir, 'feature_analysis', 'correlation_matrix.csv'), 
                                encoding='utf-8-sig')
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        high_corr_df = pd.DataFrame(high_corr_pairs)
        if len(high_corr_df) > 0:
            high_corr_df.to_csv(os.path.join(self.results_dir, 'feature_analysis', 'high_correlations.csv'), 
                               index=False, encoding='utf-8-sig')
        
        print(f"   • Total features analyzed: {len(numeric_features.columns)}")
        print(f"   • High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
        
        # Create correlation heatmap
        plt.figure(figsize=(16, 12))
        
        # Select top correlations for visualization
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("04_correlation_heatmap", 'feature_analysis')
        
        # Feature importance by target class
        print("\n🎯 FEATURE IMPORTANCE BY TARGET CLASS")
        
        feature_importance_by_target = {}
        
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            feature_importance = {}
            
            for feature in numeric_features.columns:
                feature_data = self.data[feature].dropna()
                
                # Align data
                common_indices = target_data.index.intersection(feature_data.index)
                if len(common_indices) > 10:  # Minimum sample size
                    aligned_target = target_data.loc[common_indices]
                    aligned_feature = feature_data.loc[common_indices]
                    
                    # Calculate correlation with target
                    correlation = aligned_target.corr(aligned_feature)
                    if not pd.isna(correlation):
                        feature_importance[feature] = abs(correlation)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            feature_importance_by_target[target] = feature_importance
        
        # Save feature importance analysis
        import json
        with open(os.path.join(self.results_dir, 'feature_analysis', 'feature_importance.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(feature_importance_by_target), f, indent=2, ensure_ascii=False)
        
        return feature_importance_by_target
    
    def validate_business_logic(self):
        """Validate business logic in the data"""
        print("\n" + "="*60)
        print("6️⃣ BUSINESS LOGIC VALIDATION")
        print("="*60)
        
        business_insights = {}
        
        # Financial ratio analysis
        print("\n💰 FINANCIAL RATIO ANALYSIS")
        
        # Check for common financial ratios
        financial_features = [col for col in self.feature_columns if 'financial' in col.lower()]
        
        if len(financial_features) >= 3:
            # Look for potential financial ratios
            for feature in financial_features[:5]:  # Analyze first 5
                values = self.data[feature].dropna()
                if len(values) > 0:
                    business_insights[feature] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'negative_count': (values < 0).sum(),
                        'zero_count': (values == 0).sum()
                    }
        
        # Risk progression logic
        print("\n📈 RISK PROGRESSION ANALYSIS")
        
        risk_progression = {}
        for i in range(len(self.target_columns) - 1):
            current_target = self.target_columns[i]
            next_target = self.target_columns[i + 1]
            
            # Find records with data for both targets
            current_data = self.data[current_target].dropna()
            next_data = self.data[next_target].dropna()
            
            common_indices = current_data.index.intersection(next_data.index)
            if len(common_indices) > 0:
                current_values = current_data.loc[common_indices]
                next_values = next_data.loc[common_indices]
                
                # Analyze risk progression
                risk_increases = (next_values > current_values).sum()
                risk_decreases = (next_values < current_values).sum()
                risk_stable = (next_values == current_values).sum()
                
                risk_progression[f"{current_target}_to_{next_target}"] = {
                    'total_records': len(common_indices),
                    'risk_increases': risk_increases,
                    'risk_decreases': risk_decreases,
                    'risk_stable': risk_stable,
                    'increase_rate': (risk_increases / len(common_indices)) * 100,
                    'decrease_rate': (risk_decreases / len(common_indices)) * 100
                }
        
        # Save business insights
        business_summary = {
            'financial_analysis': business_insights,
            'risk_progression': risk_progression
        }
        
        import json
        with open(os.path.join(self.results_dir, 'business_insights', 'business_logic_validation.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(business_summary), f, indent=2, ensure_ascii=False)
        
        return business_summary
    
    def investigate_data_leakage(self):
        """Investigate potential data leakage"""
        print("\n" + "="*60)
        print("7️⃣ DATA LEAKAGE INVESTIGATION")
        print("="*60)
        
        leakage_analysis = {}
        
        # Check for perfect correlations with targets
        print("\n🔍 PERFECT CORRELATION CHECK")
        
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            perfect_correlations = []
            
            for feature in self.feature_columns:
                feature_data = self.data[feature].dropna()
                
                # Align data
                common_indices = target_data.index.intersection(feature_data.index)
                if len(common_indices) > 10:
                    aligned_target = target_data.loc[common_indices]
                    aligned_feature = feature_data.loc[common_indices]
                    
                    # Check for perfect correlation
                    correlation = aligned_target.corr(aligned_feature)
                    if abs(correlation) > 0.95:  # Very high correlation threshold
                        perfect_correlations.append({
                            'feature': feature,
                            'correlation': correlation
                        })
            
            leakage_analysis[target] = perfect_correlations
        
        # Check for temporal alignment issues
        print("\n⏰ TEMPORAL ALIGNMENT CHECK")
        
        temporal_issues = []
        for feature in self.temporal_features:
            # Check if feature name suggests future information
            if any(keyword in feature.lower() for keyword in ['future', 'forecast', 'prediction']):
                temporal_issues.append({
                    'feature': feature,
                    'issue': 'Potential future information in feature name'
                })
        
        # Check for target contamination
        print("\n🎯 TARGET CONTAMINATION CHECK")
        
        target_contamination = []
        for feature in self.feature_columns:
            feature_lower = feature.lower()
            if any(target_word in feature_lower for target_word in ['risk', 'default', 'failure', 'loss']):
                target_contamination.append({
                    'feature': feature,
                    'concern': 'Feature name suggests target-related information'
                })
        
        # Create leakage summary
        leakage_summary = {
            'perfect_correlations': leakage_analysis,
            'temporal_issues': temporal_issues,
            'target_contamination': target_contamination,
            'total_concerns': len(temporal_issues) + len(target_contamination) + 
                            sum(len(corrs) for corrs in leakage_analysis.values())
        }
        
        # Save leakage analysis
        import json
        with open(os.path.join(self.results_dir, 'business_insights', 'data_leakage_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(leakage_summary), f, indent=2, ensure_ascii=False)
        
        print(f"   • Perfect correlations found: {sum(len(corrs) for corrs in leakage_analysis.values())}")
        print(f"   • Temporal alignment issues: {len(temporal_issues)}")
        print(f"   • Target contamination concerns: {len(target_contamination)}")
        print(f"   • Total concerns: {leakage_summary['total_concerns']}")
        
        return leakage_summary
    
    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        print("\n" + "="*60)
        print("8️⃣ COMPREHENSIVE REPORT GENERATION")
        print("="*60)
        
        # Create executive summary
        report = {
            'eda_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_summary': {
                'total_records': len(self.data),
                'total_features': len(self.feature_columns),
                'target_variables': len(self.target_columns),
                'temporal_features': len(self.temporal_features),
                'static_features': len(self.static_features)
            },
            'key_findings': {},
            'recommendations': []
        }
        
        # Add key findings
        print("\n📋 GENERATING KEY FINDINGS...")
        
        # Data quality findings
        missing_rate = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        report['key_findings']['data_quality'] = {
            'missing_data_rate': f"{missing_rate:.2f}%",
            'outlier_concerns': "Moderate" if missing_rate > 5 else "Low",
            'data_completeness': "Good" if missing_rate < 10 else "Needs attention"
        }
        
        # Target variable findings
        target_distributions = {}
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            class_0_rate = (target_data == 0).sum() / len(target_data) * 100
            target_distributions[target] = {
                'class_0_rate': f"{class_0_rate:.1f}%",
                'imbalance_level': "Severe" if class_0_rate > 80 else "Moderate" if class_0_rate > 60 else "Balanced"
            }
        
        report['key_findings']['target_variables'] = target_distributions
        
        # Feature relationship findings
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number])
        high_corr_count = ((numeric_features.corr().abs() > 0.8).sum().sum() - len(numeric_features.columns)) // 2
        report['key_findings']['feature_relationships'] = {
            'high_correlations': high_corr_count,
            'multicollinearity_concern': "High" if high_corr_count > 10 else "Moderate" if high_corr_count > 5 else "Low"
        }
        
        # Generate recommendations
        print("\n💡 GENERATING RECOMMENDATIONS...")
        
        recommendations = []
        
        # Data quality recommendations
        if missing_rate > 5:
            recommendations.append("Implement sophisticated missing value imputation strategies")
        
        # Class imbalance recommendations
        if any(target_distributions[t]['imbalance_level'] == 'Severe' for t in target_distributions):
            recommendations.append("Apply class imbalance handling techniques (SMOTE, class weights)")
        
        # Feature selection recommendations
        if high_corr_count > 5:
            recommendations.append("Implement feature selection to address multicollinearity")
        
        # Temporal validation recommendations
        if len(self.temporal_features) > 0:
            recommendations.append("Use temporal train/test splits to prevent optimistic bias")
        
        # Model architecture recommendations
        recommendations.append("Consider ensemble methods for improved robustness")
        recommendations.append("Implement cost-sensitive learning for high-risk detection")
        
        report['recommendations'] = recommendations
        
        # Save comprehensive report
        import json
        with open(os.path.join(self.results_dir, 'comprehensive_eda_report.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(report), f, indent=2, ensure_ascii=False)
        
        # Print executive summary
        print(f"\n📊 EXECUTIVE SUMMARY - STEP 3 EDA")
        print("=" * 60)
        print(f"📅 Analysis Date: {report['eda_date']}")
        print(f"📊 Dataset: {report['dataset_summary']['total_records']:,} records, {report['dataset_summary']['total_features']} features")
        print(f"🎯 Targets: {report['dataset_summary']['target_variables']} risk prediction horizons")
        print(f"⏰ Temporal Features: {report['dataset_summary']['temporal_features']}")
        
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"• Data Quality: {report['key_findings']['data_quality']['data_completeness']}")
        print(f"• Class Imbalance: {target_distributions[self.target_columns[0]]['imbalance_level']} in Year1")
        print(f"• Multicollinearity: {report['key_findings']['feature_relationships']['multicollinearity_concern']} concern")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")
        
        return report
    
    def run_step3_eda(self):
        """Execute complete Step 3 EDA pipeline"""
        print("🔍 EXECUTING STEP 3: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Data quality analysis
            self.analyze_data_quality()
            
            # Step 3: Temporal pattern analysis
            self.analyze_temporal_patterns()
            
            # Step 4: Target variable analysis
            self.analyze_target_variables()
            
            # Step 5: Feature relationship analysis
            self.analyze_feature_relationships()
            
            # Step 6: Business logic validation
            self.validate_business_logic()
            
            # Step 7: Data leakage investigation
            self.investigate_data_leakage()
            
            # Step 8: Generate comprehensive report
            self.generate_comprehensive_report()
            
            print("\n🎉 STEP 3 COMPLETED SUCCESSFULLY!")
            print("✅ Comprehensive data understanding achieved")
            print("✅ Business insights generated")
            print("✅ Ready for informed modeling decisions")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 3 FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Data Quality:")
        print(f"      • missing_value_analysis.csv")
        print(f"      • outlier_analysis.csv")
        print(f"      • quality_summary.json")
        print(f"   📈 Temporal Analysis:")
        print(f"      • temporal_correlations.json")
        print(f"   🎯 Target Analysis:")
        print(f"      • target_analysis.json")
        print(f"   🔗 Feature Analysis:")
        print(f"      • correlation_matrix.csv")
        print(f"      • high_correlations.csv")
        print(f"      • feature_importance.json")
        print(f"   💼 Business Insights:")
        print(f"      • business_logic_validation.json")
        print(f"      • data_leakage_analysis.json")
        print(f"   📋 Comprehensive Report:")
        print(f"      • comprehensive_eda_report.json")
        print(f"   📊 Visualizations:")
        print(f"      • 4 comprehensive analysis plots")
        print(f"\n💡 Use these insights to inform Steps 4-7!")


def get_step3_config():
    """Configuration for Step 3 EDA"""
    return {
        'data_path': 'dataset/credit_risk_dataset.csv',
        'exclude_columns': [
            '사업자등록번호',
            '대상자명',
            '대상자등록이력일시',
            '대상자기본주소',
            '청약번호',
            '보험청약일자',
            '청약상태코드',
            '수출자대상자번호',
            '특별출연협약코드',
            '업종코드1'
        ],
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4']
    }


# Main execution
if __name__ == "__main__":
    
    print("🔍 Starting XGBoost Risk Prediction Model - Step 3")
    print("="*60)
    
    # Get configuration
    config = get_step3_config()
    
    # Create and run EDA framework
    eda = ComprehensiveEDA(config)
    eda.run_step3_eda()
    
    print("\n🏁 Step 3 execution completed!")
    print("Ready to proceed to Step 4: Temporal Train/Test Split") 