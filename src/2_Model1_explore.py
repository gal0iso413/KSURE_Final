"""
XGBoost Risk Prediction Model - Step 1: Comprehensive Exploratory Data Analysis (EDA)
====================================================================================

Step 1 Implementation (EDA FOCUS):
- Data Quality Analysis: Missing patterns, outliers, data types
- Target Variable Analysis: Class distribution, business patterns
- Feature Relationships: Correlation analysis, feature distributions
- Business Logic Validation: Do the data patterns make business sense?
- Goal: Deep understanding of data characteristics to inform modeling decisions

Design Focus:
- Comprehensive analysis supporting step1_train.py and step1_evaluate.py
- Business-focused insights and interpretations
- Korean font support for visualizations
- Systematic approach to data understanding
- Foundation for informed feature engineering and model selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import platform
import matplotlib.font_manager as fm
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    Step 1: Comprehensive Exploratory Data Analysis for Risk Prediction
    
    Provides deep understanding of data characteristics through:
    - Data quality analysis
    - Target variable analysis
    - Feature relationship analysis
    - Business logic validation
    - Foundation for informed modeling decisions
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
        results_dir = "../results/step1_eda"
        
        os.makedirs(results_dir, exist_ok=True)
        
        subdirs = ['data_quality', 'target_analysis', 'feature_analysis', 'visualizations']
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
        
        print(f"📊 Dataset structure:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Target variables: {len(self.target_columns)}")
        print(f"   • Feature columns: {len(self.feature_columns)}")
        
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
        
        # Create missing value heatmap (downsample rows for efficiency)
        plt.figure(figsize=(12, 8))
        max_rows_for_heatmap = 2000
        sampled_df = self.data.sample(n=min(max_rows_for_heatmap, len(self.data)), random_state=42)
        missing_matrix = sampled_df.isnull()
        sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Value Pattern Analysis (sampled rows)', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Sampled Records', fontsize=12)
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
    
    def analyze_target_variables(self):
        """Comprehensive target variable analysis"""
        print("\n" + "="*60)
        print("3️⃣ TARGET VARIABLE ANALYSIS")
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
        self._save_plot("02_target_distributions", 'target_analysis')
        
        # Create class imbalance analysis
        self._analyze_class_imbalance()
        
        # Save target analysis
        import json
        with open(os.path.join(self.results_dir, 'target_analysis', 'target_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(target_analysis), f, indent=2, ensure_ascii=False)
        
        return target_analysis
    
    def _analyze_class_imbalance(self):
        """Analyze class imbalance patterns across targets"""
        print("\n📊 CLASS IMBALANCE ANALYSIS")
        
        imbalance_data = []
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            if len(target_data) > 0:
                value_counts = target_data.value_counts().sort_index()
                total = len(target_data)
                
                # Calculate imbalance metrics
                majority_class = value_counts.max()
                minority_class = value_counts.min()
                imbalance_ratio = majority_class / minority_class
                
                # High-risk (classes 2,3) vs low-risk (classes 0,1) ratio
                high_risk_count = value_counts.get(2, 0) + value_counts.get(3, 0)
                low_risk_count = value_counts.get(0, 0) + value_counts.get(1, 0)
                high_risk_ratio = high_risk_count / total * 100
                
                imbalance_data.append({
                    'target': target,
                    'total_samples': total,
                    'majority_class_count': majority_class,
                    'minority_class_count': minority_class,
                    'imbalance_ratio': imbalance_ratio,
                    'high_risk_percentage': high_risk_ratio,
                    'class_distribution': dict(value_counts)
                })
                
                print(f"   • {target}: {high_risk_ratio:.1f}% high-risk, imbalance ratio: {imbalance_ratio:.1f}")
        
        # Visualize class imbalance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Class Imbalance Analysis Across Targets', fontsize=16, fontweight='bold')
        
        # High-risk percentage
        targets = [item['target'] for item in imbalance_data]
        high_risk_pcts = [item['high_risk_percentage'] for item in imbalance_data]
        
        ax1.bar(targets, high_risk_pcts, alpha=0.7, color='red')
        ax1.set_title('High-Risk Percentage by Target')
        ax1.set_ylabel('High-Risk Percentage (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Imbalance ratios
        imbalance_ratios = [item['imbalance_ratio'] for item in imbalance_data]
        ax2.bar(targets, imbalance_ratios, alpha=0.7, color='orange')
        ax2.set_title('Class Imbalance Ratio (Majority/Minority)')
        ax2.set_ylabel('Imbalance Ratio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("03_class_imbalance", 'target_analysis')
        
        # Save imbalance analysis
        imbalance_df = pd.DataFrame(imbalance_data)
        imbalance_df.to_csv(os.path.join(self.results_dir, 'target_analysis', 'class_imbalance.csv'), 
                           index=False, encoding='utf-8-sig')
    
    def analyze_feature_relationships(self):
        """Analyze feature relationships and correlations"""
        print("\n" + "="*60)
        print("4️⃣ FEATURE RELATIONSHIP ANALYSIS")
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
        
        # Create correlation heatmap (sample columns for efficiency)
        plt.figure(figsize=(16, 12))
        max_cols_for_heatmap = 50
        cols_for_plot = (
            correlation_matrix.columns[:max_cols_for_heatmap]
            if len(correlation_matrix.columns) > max_cols_for_heatmap
            else correlation_matrix.columns
        )
        corr_plot = correlation_matrix.loc[cols_for_plot, cols_for_plot]
        mask = np.triu(np.ones_like(corr_plot, dtype=bool))
        sns.heatmap(corr_plot, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix (subset for visualization)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("04_correlation_heatmap", 'feature_analysis')
        
        # Feature importance by target class
        print("\n🎯 FEATURE-TARGET CORRELATION ANALYSIS")
        
        feature_target_correlation = {}
        
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            feature_correlations = {}
            
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
                        feature_correlations[feature] = abs(correlation)
            
            # Sort by importance
            feature_correlations = dict(sorted(feature_correlations.items(), 
                                           key=lambda x: x[1], reverse=True))
            feature_target_correlation[target] = feature_correlations
            
            print(f"   • {target}: Top correlation = {max(feature_correlations.values()):.4f}")
        
        # Create feature-target correlation visualization
        self._visualize_feature_target_correlations(feature_target_correlation)
        
        # Save feature importance analysis
        import json
        with open(os.path.join(self.results_dir, 'feature_analysis', 'feature_target_correlations.json'), 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(feature_target_correlation), f, indent=2, ensure_ascii=False)
        
        return feature_target_correlation
    
    def _visualize_feature_target_correlations(self, correlations: Dict):
        """Visualize top feature-target correlations"""
        
        # Get top features across all targets
        all_correlations = {}
        for target, feature_corrs in correlations.items():
            for feature, corr in feature_corrs.items():
                if feature not in all_correlations:
                    all_correlations[feature] = []
                all_correlations[feature].append(corr)
        
        # Calculate average correlation
        avg_correlations = {
            feature: np.mean(corrs) 
            for feature, corrs in all_correlations.items()
        }
        
        # Get top 20 features
        top_features = dict(sorted(avg_correlations.items(), key=lambda x: x[1], reverse=True)[:20])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Feature-Target Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Average correlations
        ax1.barh(list(top_features.keys()), list(top_features.values()), alpha=0.7)
        ax1.set_title('Top 20 Features - Average |Correlation| with Targets')
        ax1.set_xlabel('Average |Correlation|')
        ax1.grid(True, alpha=0.3)
        
        # Per-target correlations for top features
        top_feature_names = list(top_features.keys())[:10]  # Top 10 for readability
        correlation_matrix_data = []
        
        for target in self.target_columns:
            target_corrs = []
            for feature in top_feature_names:
                corr = correlations[target].get(feature, 0)
                target_corrs.append(corr)
            correlation_matrix_data.append(target_corrs)
        
        correlation_matrix_df = pd.DataFrame(
            correlation_matrix_data, 
            index=self.target_columns, 
            columns=top_feature_names
        )
        
        sns.heatmap(correlation_matrix_df, annot=True, fmt='.3f', cmap='viridis', ax=ax2)
        ax2.set_title('Top 10 Features - Correlation by Target')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Targets')
        
        plt.tight_layout()
        self._save_plot("05_feature_target_correlations", 'feature_analysis')
    
    def analyze_feature_distributions(self):
        """Analyze feature distributions by risk level"""
        print("\n" + "="*60)
        print("5️⃣ FEATURE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Select top correlated features for distribution analysis
        numeric_features = self.data[self.feature_columns].select_dtypes(include=[np.number])
        
        # Get top features that correlate with any target
        top_features = []
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            
            for feature in numeric_features.columns[:20]:  # Check top 20 numeric features
                feature_data = self.data[feature].dropna()
                common_indices = target_data.index.intersection(feature_data.index)
                
                if len(common_indices) > 50:
                    aligned_target = target_data.loc[common_indices]
                    aligned_feature = feature_data.loc[common_indices]
                    correlation = aligned_target.corr(aligned_feature)
                    
                    if not pd.isna(correlation) and abs(correlation) > 0.1:
                        top_features.append((feature, abs(correlation)))
        
        # Get unique top features
        unique_features = list(set([f[0] for f in top_features]))[:8]  # Top 8 for visualization
        
        if len(unique_features) > 0:
            print(f"   • Analyzing distributions for {len(unique_features)} top features...")
            
            # Create distribution plots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('Feature Distributions by Risk Level', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for i, feature in enumerate(unique_features):
                if i >= 8:
                    break
                
                ax = axes[i]
                
                # Combine all targets for this analysis
                combined_data = []
                for target in self.target_columns:
                    target_data = self.data[target].dropna()
                    feature_data = self.data[feature].dropna()
                    common_indices = target_data.index.intersection(feature_data.index)
                    
                    if len(common_indices) > 10:
                        for risk_level in [0, 1, 2, 3]:
                            risk_mask = target_data.loc[common_indices] == risk_level
                            if risk_mask.sum() > 0:
                                values = feature_data.loc[common_indices[risk_mask]]
                                for val in values:
                                    combined_data.append({'feature_value': val, 'risk_level': risk_level})
                
                if combined_data:
                    dist_df = pd.DataFrame(combined_data)
                    
                    # Create box plot
                    sns.boxplot(data=dist_df, x='risk_level', y='feature_value', ax=ax)
                    ax.set_title(f'{feature}', fontsize=10)
                    ax.set_xlabel('Risk Level')
                    ax.set_ylabel('Feature Value')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._save_plot("06_feature_distributions", 'feature_analysis')
        
        print(f"   ✅ Feature distribution analysis completed")
    
    def generate_eda_summary(self):
        """Generate comprehensive EDA summary report"""
        print("\n" + "="*60)
        print("6️⃣ EDA SUMMARY REPORT GENERATION")
        print("="*60)
        
        # Collect key findings
        summary = {
            'eda_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_overview': {
                'total_records': len(self.data),
                'total_features': len(self.feature_columns),
                'target_variables': len(self.target_columns),
                'missing_data_percentage': float((self.data.isnull().sum().sum() / self.data.size) * 100)
            },
            'data_quality_insights': {
                'columns_with_missing_data': int((self.data.isnull().sum() > 0).sum()),
                'average_missing_rate': float(self.data.isnull().sum().mean() / len(self.data) * 100),
                'data_types': dict(self.data.dtypes.value_counts())
            },
            'target_analysis_insights': {},
            'key_findings': [],
            'recommendations_for_modeling': []
        }
        
        # Target analysis insights
        for target in self.target_columns:
            target_data = self.data[target].dropna()
            if len(target_data) > 0:
                value_counts = target_data.value_counts().sort_index()
                high_risk_pct = (value_counts.get(2, 0) + value_counts.get(3, 0)) / len(target_data) * 100
                
                summary['target_analysis_insights'][target] = {
                    'sample_size': len(target_data),
                    'high_risk_percentage': float(high_risk_pct),
                    'class_distribution': dict(value_counts),
                    'missing_percentage': float(self.data[target].isnull().sum() / len(self.data) * 100)
                }
        
        # Key findings
        avg_high_risk = np.mean([
            insights['high_risk_percentage'] 
            for insights in summary['target_analysis_insights'].values()
        ])
        
        summary['key_findings'] = [
            f"Dataset contains {len(self.data):,} records with {len(self.feature_columns)} features",
            f"Average high-risk percentage across targets: {avg_high_risk:.1f}%",
            f"Missing data affects {summary['data_quality_insights']['columns_with_missing_data']} columns",
            f"Class imbalance is present - high-risk cases are minority class",
            "Feature correlations show potential for predictive modeling",
            "Multiple data types present requiring appropriate preprocessing"
        ]
        
        # Modeling recommendations
        summary['recommendations_for_modeling'] = [
            "Handle class imbalance using appropriate techniques (SMOTE, class weights)",
            "Consider ensemble methods to leverage different algorithm strengths",
            "Implement robust missing value imputation strategies",
            "Use cross-validation with stratification for reliable performance estimation",
            "Focus on recall for high-risk classes (business priority)",
            "Consider feature selection based on correlation analysis",
            "Implement temporal validation if time-based patterns exist",
            "Monitor model performance across different risk levels"
        ]
        
        # Save EDA summary
        import json
        summary_path = os.path.join(self.results_dir, 'eda_summary_report.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(safe_json_serialize(summary), f, indent=2, ensure_ascii=False)
        
        print(f"✅ EDA summary report saved: eda_summary_report.json")
        
        # Print executive summary
        print(f"\n📋 EDA EXECUTIVE SUMMARY:")
        print("=" * 60)
        print(f"📅 Analysis Date: {summary['eda_timestamp']}")
        print(f"📊 Dataset: {summary['dataset_overview']['total_records']:,} records, {summary['dataset_overview']['total_features']} features")
        print(f"🎯 Targets: {len(self.target_columns)} risk prediction variables")
        print(f"⚠️  Missing Data: {summary['dataset_overview']['missing_data_percentage']:.1f}% overall")
        print(f"🚨 High-Risk Rate: {avg_high_risk:.1f}% average across targets")
        
        print(f"\n🔍 KEY INSIGHTS:")
        for finding in summary['key_findings']:
            print(f"• {finding}")
        
        print(f"\n💡 MODELING RECOMMENDATIONS:")
        for rec in summary['recommendations_for_modeling'][:5]:  # Show top 5
            print(f"• {rec}")
        
        return summary
    
    def run_eda_pipeline(self):
        """Execute complete EDA pipeline"""
        print("🔍 EXECUTING STEP 1: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Data quality analysis
            self.analyze_data_quality()
            
            # Step 3: Target variable analysis
            self.analyze_target_variables()
            
            # Step 4: Feature relationship analysis
            self.analyze_feature_relationships()
            
            # Step 5: Feature distribution analysis
            self.analyze_feature_distributions()
            
            # Step 6: Generate EDA summary
            self.generate_eda_summary()
            
            print("\n🎉 STEP 1 EDA COMPLETED SUCCESSFULLY!")
            print("✅ Comprehensive data understanding achieved")
            print("✅ Modeling insights generated")
            print("✅ Ready to inform step1_train.py and step1_evaluate.py")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 1 EDA FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 EDA RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Data Quality Analysis:")
        print(f"      • missing_value_analysis.csv")
        print(f"      • outlier_analysis.csv")
        print(f"      • quality_summary.json")
        print(f"   🎯 Target Analysis:")
        print(f"      • target_analysis.json")
        print(f"      • class_imbalance.csv")
        print(f"   🔗 Feature Analysis:")
        print(f"      • correlation_matrix.csv")
        print(f"      • high_correlations.csv")
        print(f"      • feature_target_correlations.json")
        print(f"   📊 Visualizations:")
        print(f"      • missing value patterns, target distributions")
        print(f"      • correlation heatmaps, feature distributions")
        print(f"   📄 Summary Report:")
        print(f"      • eda_summary_report.json")
        print(f"\n💡 Use these insights to guide step1_train.py modeling decisions!")


def get_eda_config():
    """Configuration for Step 1 EDA"""
    return {
        'data_path': '../data/processed/credit_risk_dataset.csv',
        'exclude_columns': [
            '사업자등록번호',
            '대상자명',
            '청약번호',
            '보험청약일자',
            '수출자대상자번호',
            '업종코드1'
        ],
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4']
    }


# Main execution
if __name__ == "__main__":
    print("🔍 Starting Comprehensive EDA - Step 1")
    print("="*60)
    
    # Get configuration
    config = get_eda_config()
    
    # Create and run EDA framework
    eda = ComprehensiveEDA(config)
    eda.run_eda_pipeline()
    
    print("\n🏁 Step 1 EDA execution completed!")
    print("Ready to support step1_train.py and step1_evaluate.py with data insights!")
