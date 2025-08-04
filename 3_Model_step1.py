"""
XGBoost Risk Prediction Model - Step 1: Simple Baseline Model
============================================================

Step 1 Implementation:
- Load data from 1_Dataset.py output
- Basic data exploration (shape, missing values, target distribution)
- Create 4 separate XGBoost regression models with default parameters
- Use simple train/test split (80/20) without considering dates
- Goal: Establish working pipeline and baseline performance

Design Decisions:
- 4 Separate Models: One for each risk_year (1,2,3,4)
- Regression with Rounding: Treat as continuous then round to preserve ordinality
- Native Missing Handling: Let XGBoost handle missing X variables
- Target-Specific Filtering: Each model uses only rows with valid data for its specific target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings
import os
import platform
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# Configure matplotlib for non-interactive backend (no GUI needed)
import matplotlib
matplotlib.use('Agg')

# Configure Korean font for matplotlib
def setup_korean_font():
    """Set up Korean font for matplotlib visualizations"""
    system = platform.system()
    
    if system == "Windows":
        # Common Korean fonts on Windows
        korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Gulim', 'Dotum']
    elif system == "Darwin":  # macOS
        korean_fonts = ['AppleGothic', 'NanumGothic']
    else:  # Linux
        korean_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum']
    
    # Try to find and set Korean font
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
        # Fallback: try to use any available font that supports Korean
        print("⚠️  Preferred Korean fonts not found. Trying fallback options...")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        print("📝 Using fallback font. Korean characters may not display correctly.")
    
    # Ensure minus signs display correctly
    plt.rcParams['axes.unicode_minus'] = False
    
    return korean_font

# Set up Korean font
setup_korean_font()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class BaselineRiskModel:
    """
    Step 1: Simple Baseline XGBoost Model for Risk Prediction
    
    Creates 4 separate regression models for predicting risk at years 1-4.
    Focuses on establishing working pipeline and baseline performance.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize baseline model with configuration
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}
        self.models = {}
        self.predictions = {}
        self.feature_columns = None
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        self.korean_font = setup_korean_font()
        
        print("🚀 Baseline Risk Model Initialized")
        print(f"📊 Target variables: {config['target_columns']}")
        print(f"🎯 Test size: {config['test_size']}")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def _create_results_directory(self) -> str:
        """
        Create results directory for saving outputs
        
        Returns:
            Path to created results directory
        """
        results_dir = "result/step1_baseline"
        
        # Create main results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        subdirs = ['visualizations', 'models', 'metrics']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir
    
    def _save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
        """
        Save current plot to results directory
        
        Args:
            filename: Name of the file (without extension)
            dpi: Resolution for PNG files
            bbox_inches: Bounding box setting
            
        Returns:
            Full path to saved file
        """
        # Save as PNG only
        png_path = os.path.join(self.results_dir, 'visualizations', f"{filename}.png")
        
        plt.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        plt.close()  # Close figure to free memory
        
        print(f"  💾 Saved: {filename}.png")
        return png_path
    
    def load_and_explore_data(self) -> pd.DataFrame:
        """
        Load dataset and perform basic exploration
        
        Returns:
            Loaded dataframe
        """
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING & EXPLORATION")
        print("="*60)
        
        # Load data
        try:
            self.data = pd.read_csv(self.config['data_path'])
            print(f"✅ Data loaded successfully")
            print(f"📊 Shape: {self.data.shape}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
        
        # Basic exploration
        print(f"\n📈 BASIC STATISTICS:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Total columns: {len(self.data.columns):,}")
        
        # Missing values overview
        print(f"\n🔍 MISSING VALUES OVERVIEW:")
        missing_counts = self.data.isnull().sum()
        total_missing = missing_counts.sum()
        print(f"   • Total missing values: {total_missing:,}")
        print(f"   • Columns with missing data: {(missing_counts > 0).sum()}")
        
        # Target variable analysis
        print(f"\n🎯 TARGET VARIABLES ANALYSIS:")
        target_cols = self.config['target_columns']
        
        for target in target_cols:
            if target in self.data.columns:
                non_null_count = self.data[target].notna().sum()
                unique_values = sorted(self.data[target].dropna().unique())
                value_counts = self.data[target].value_counts().sort_index()
                
                print(f"   • {target}:")
                print(f"     - Non-null records: {non_null_count:,}")
                print(f"     - Unique values: {unique_values}")
                print(f"     - Distribution: {dict(value_counts)}")
            else:
                print(f"   ❌ {target} not found in dataset")
        
        # Feature type analysis
        print(f"\n📊 FEATURE TYPES ANALYSIS:")
        feature_types = {}
        for col in self.data.columns:
            if not col.startswith('risk_year'):
                prefix = col.split('_')[0] if '_' in col else 'other'
                feature_types[prefix] = feature_types.get(prefix, 0) + 1
        
        for ftype, count in sorted(feature_types.items()):
            print(f"   • {ftype}: {count} features")
        
        return self.data
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Preprocess data for training
        
        Returns:
            X_train, X_test, y_train_dict, y_test_dict
        """
        print("\n" + "="*60)
        print("2️⃣ DATA PREPROCESSING")
        print("="*60)
        
        # Apply column exclusions (but keep all rows - filtering will be done per model)
        target_cols = self.config['target_columns']
        processed_data = self.data.copy()
        
        exclude_cols = self.config.get('exclude_columns', [])
        if exclude_cols:
            print(f"🚫 Excluding {len(exclude_cols)} specified columns...")
            processed_data = processed_data.drop(columns=[col for col in exclude_cols if col in processed_data.columns])
        
        # Separate features and targets
        feature_cols = [col for col in processed_data.columns if not col.startswith('risk_year')]
        target_data = {target: processed_data[target].values for target in target_cols if target in processed_data.columns}
        
        X = processed_data[feature_cols]
        
        print(f"📊 Dataset structure (before target-specific filtering):")
        print(f"   • Total records: {len(processed_data):,}")
        print(f"   • Feature columns: {len(feature_cols)}")
        print(f"   • Target variables: {len(target_data)}")
        
        # Show target availability statistics
        print(f"\n🎯 Target variable availability:")
        for target in target_cols:
            if target in processed_data.columns:
                non_null_count = processed_data[target].notna().sum()
                print(f"   • {target}: {non_null_count:,} records ({non_null_count/len(processed_data)*100:.1f}%)")
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Train/test split on full dataset
        print(f"\n🔄 Creating train/test split ({int((1-self.config['test_size'])*100)}/{int(self.config['test_size']*100)})...")
        
        X_train, X_test = train_test_split(
            X, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            shuffle=True
        )
        
        # Split each target variable (keeping all values including NaN)
        y_train_dict = {}
        y_test_dict = {}
        
        for target_name, target_values in target_data.items():
            y_train_target, y_test_target = train_test_split(
                target_values,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                shuffle=True
            )
            y_train_dict[target_name] = y_train_target
            y_test_dict[target_name] = y_test_target
        
        print(f"   • Training records: {len(X_train):,}")
        print(f"   • Test records: {len(X_test):,}")
        print("   📝 Note: Target-specific filtering will be applied during model training")
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_dict
        self.y_test = y_test_dict
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train_models(self) -> Dict:
        """
        Train separate XGBoost regression models for each target
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("3️⃣ MODEL TRAINING")
        print("="*60)
        
        models = {}
        xgb_params = self.config.get('xgb_params', {})
        
        # Default parameters for baseline model
        default_params = {
            'objective': 'reg:squarederror',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
        default_params.update(xgb_params)
        
        print(f"🔧 XGBoost parameters: {default_params}")
        print(f"📈 Training {len(self.y_train)} separate models with target-specific filtering...")
        
        for target_name, y_train_values in self.y_train.items():
            print(f"\n   🎯 Training model for {target_name}...")
            
            try:
                # Filter training data for this specific target (remove NaN values)
                valid_mask = ~pd.isna(y_train_values)
                X_train_filtered = self.X_train[valid_mask]
                y_train_filtered = y_train_values[valid_mask]
                
                # Skip if no valid data
                if len(y_train_filtered) == 0:
                    print(f"      ❌ No valid training data for {target_name}")
                    continue
                
                print(f"      📊 Training samples: {len(y_train_filtered):,} (filtered from {len(y_train_values):,})")
                
                # Create and train XGBoost model
                model = xgb.XGBRegressor(**default_params)
                model.fit(X_train_filtered, y_train_filtered)
                
                models[target_name] = model
                
                # Quick training summary
                train_pred = model.predict(X_train_filtered)
                train_mae = mean_absolute_error(y_train_filtered, train_pred)
                
                print(f"      ✅ Model trained successfully")
                print(f"      📊 Training MAE: {train_mae:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error training {target_name}: {e}")
                continue
        
        self.models = models
        print(f"\n✅ Training completed: {len(models)} models ready")
        
        return models
    
    def predict_and_evaluate(self) -> Dict:
        """
        Generate predictions and evaluate model performance
        
        Returns:
            Dictionary of predictions and metrics
        """
        print("\n" + "="*60)
        print("4️⃣ PREDICTION & EVALUATION")
        print("="*60)
        
        predictions = {}
        metrics = {}
        
        print(f"🔮 Generating predictions for {len(self.models)} models...")
        
        for target_name, model in self.models.items():
            print(f"\n   📊 Evaluating {target_name}...")
            
            try:
                # Filter test data for this specific target (remove NaN values)
                y_test_values = self.y_test[target_name]
                valid_mask = ~pd.isna(y_test_values)
                X_test_filtered = self.X_test[valid_mask]
                y_test_filtered = y_test_values[valid_mask]
                
                # Skip if no valid test data
                if len(y_test_filtered) == 0:
                    print(f"      ❌ No valid test data for {target_name}")
                    continue
                
                print(f"      📊 Test samples: {len(y_test_filtered):,} (filtered from {len(y_test_values):,})")
                
                # Generate predictions
                y_pred_raw = model.predict(X_test_filtered)
                
                # Round to nearest integer and clip to valid range [0, 3]
                y_pred_rounded = np.clip(np.round(y_pred_raw), 0, 3).astype(int)
                
                # Store predictions
                predictions[target_name] = {
                    'raw': y_pred_raw,
                    'rounded': y_pred_rounded,
                    'actual': y_test_filtered
                }
                
                # Calculate metrics
                mae = mean_absolute_error(y_test_filtered, y_pred_raw)
                mae_rounded = mean_absolute_error(y_test_filtered, y_pred_rounded)
                accuracy = accuracy_score(y_test_filtered, y_pred_rounded)
                
                # Store metrics
                metrics[target_name] = {
                    'mae_raw': mae,
                    'mae_rounded': mae_rounded,
                    'accuracy': accuracy,
                    'prediction_range': f"[{y_pred_raw.min():.2f}, {y_pred_raw.max():.2f}]"
                }
                
                print(f"      📈 MAE (raw): {mae:.4f}")
                print(f"      📈 MAE (rounded): {mae_rounded:.4f}")
                print(f"      🎯 Accuracy: {accuracy:.4f}")
                print(f"      📊 Prediction range: {metrics[target_name]['prediction_range']}")
                
            except Exception as e:
                print(f"      ❌ Error evaluating {target_name}: {e}")
                continue
        
        self.predictions = predictions
        
        # Print overall summary
        print(f"\n📋 EVALUATION SUMMARY:")
        if metrics:
            avg_mae = np.mean([m['mae_rounded'] for m in metrics.values()])
            avg_accuracy = np.mean([m['accuracy'] for m in metrics.values()])
            print(f"   • Average MAE (rounded): {avg_mae:.4f}")
            print(f"   • Average Accuracy: {avg_accuracy:.4f}")
        
        return predictions
    
    def create_visualizations(self):
        """
        Create basic visualizations for model performance
        """
        print("\n" + "="*60)
        print("5️⃣ VISUALIZATION & SUMMARY")
        print("="*60)
        
        if not self.predictions:
            print("❌ No predictions available for visualization")
            return
        
        # Create figure with subplots
        n_targets = len(self.predictions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baseline Model Performance - Step 1', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # 1. Prediction vs Actual scatter plots
        for i, (target_name, pred_data) in enumerate(self.predictions.items()):
            if i >= 4:  # Only show first 4
                break
                
            ax = axes[i]
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val, max_val = 0, 3
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Formatting
            ax.set_xlabel('Actual Risk Level')
            ax.set_ylabel('Predicted Risk Level')
            ax.set_title(f'{target_name} - Predictions vs Actual')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 3.5)
            
            # Add metrics text
            mae = mean_absolute_error(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'MAE: {mae:.3f}\nAcc: {acc:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        self._save_plot("01_prediction_vs_actual")
        
        # 2. Feature Importance Analysis
        self._plot_feature_importance()
        
        # 3. Performance Summary Table
        self._print_performance_summary()
    
    def _plot_feature_importance(self):
        """Plot feature importance for all models"""
        if not self.models:
            return
            
        print("\n📊 Feature Importance Analysis...")
        
        # Collect feature importance from all models
        importance_data = {}
        for target_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[target_name] = model.feature_importances_
        
        if not importance_data:
            print("   ❌ No feature importance data available")
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame(importance_data, index=self.feature_columns)
        
        # Calculate average importance across all models
        importance_df['avg_importance'] = importance_df.mean(axis=1)
        
        # Get top 20 most important features
        top_features = importance_df.nlargest(20, 'avg_importance')
        
        # Plot with improved layout for Korean text
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Average importance (horizontal bar chart)
        top_features['avg_importance'].plot(kind='barh', ax=ax1)
        ax1.set_title('Top 20 Features - Average Importance')
        ax1.set_xlabel('Importance Score')
        
        # Importance by target (horizontal bar chart to avoid x-axis label overlap)
        target_cols = [col for col in top_features.columns if col != 'avg_importance']
        if target_cols:
            # Transpose data for horizontal plotting
            plot_data = top_features[target_cols].T
            plot_data.plot(kind='barh', ax=ax2)
            ax2.set_title('Feature Importance by Target Variable')
            ax2.set_xlabel('Importance Score')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.85, top=0.9)
        self._save_plot("02_feature_importance")
        
        # Print top features
        print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(top_features['avg_importance'].head(10).items(), 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    def _print_performance_summary(self):
        """Print comprehensive performance summary and save to files"""
        if not self.predictions:
            return
            
        print(f"\n📊 COMPREHENSIVE PERFORMANCE SUMMARY")
        print("-" * 60)
        
        # Create summary table
        summary_data = []
        for target_name, pred_data in self.predictions.items():
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            
            # Calculate detailed metrics
            mae = mean_absolute_error(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Class distribution
            true_dist = pd.Series(y_true).value_counts().sort_index()
            pred_dist = pd.Series(y_pred).value_counts().sort_index()
            
            summary_data.append({
                'Target': target_name,
                'MAE': f"{mae:.4f}",
                'Accuracy': f"{accuracy:.4f}",
                'True_Dist': dict(true_dist),
                'Pred_Dist': dict(pred_dist)
            })
        
        # Print summary table
        for data in summary_data:
            print(f"\n🎯 {data['Target']}:")
            print(f"   • MAE: {data['MAE']}")
            print(f"   • Accuracy: {data['Accuracy']}")
            print(f"   • Actual distribution: {data['True_Dist']}")
            print(f"   • Predicted distribution: {data['Pred_Dist']}")
        
        # Save performance summary to files
        self._save_performance_metrics(summary_data)
    
    def _save_performance_metrics(self, summary_data: List[Dict]):
        """
        Save performance metrics to CSV and JSON files
        
        Args:
            summary_data: List of performance data dictionaries
        """
        print("\n💾 Saving performance metrics to files...")
        
        # Prepare data for CSV (flatten dictionaries)
        csv_data = []
        for data in summary_data:
            csv_row = {
                'Target': data['Target'],
                'MAE': float(data['MAE']),
                'Accuracy': float(data['Accuracy']),
            }
            # Add distribution data
            for level in [0, 1, 2, 3]:
                csv_row[f'True_Risk_{level}'] = data['True_Dist'].get(level, 0)
                csv_row[f'Pred_Risk_{level}'] = data['Pred_Dist'].get(level, 0)
            csv_data.append(csv_row)
        
        # Save CSV
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.results_dir, 'metrics', 'step1_performance_summary.csv')
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  📊 Performance summary saved: step1_performance_summary.csv")
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, 'metrics', 'step1_detailed_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"  📄 Detailed results saved: step1_detailed_results.json")
        
        # Save model files
        self._save_models()
    
    def _save_models(self):
        """Save trained models to files"""
        print("\n💾 Saving trained models...")
        
        for target_name, model in self.models.items():
            model_path = os.path.join(self.results_dir, 'models', f'{target_name}_model.json')
            model.save_model(model_path)
            print(f"  🤖 Model saved: {target_name}_model.json")
    
    def run_step1_pipeline(self):
        """
        Execute complete Step 1 pipeline
        """
        print("🏗️ EXECUTING STEP 1: SIMPLE BASELINE MODEL")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Predict and evaluate
            self.predict_and_evaluate()
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            print("\n🎉 STEP 1 COMPLETED SUCCESSFULLY!")
            print("✅ Baseline performance established")
            print("✅ Working pipeline ready for Step 2")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 1 FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Visualizations:")
        print(f"      • 01_prediction_vs_actual.png")
        print(f"      • 02_feature_importance.png")
        print(f"   📈 Metrics:")
        print(f"      • step1_performance_summary.csv")
        print(f"      • step1_detailed_results.json")
        print(f"   🤖 Models:")
        for target in self.config['target_columns']:
            print(f"      • {target}_model.json")
        print(f"\n💡 Use these files for offline analysis and reporting!")


def get_step1_config():
    """
    Configuration for Step 1 baseline model
    """
    return {
        'data_path': 'dataset/credit_risk_dataset.csv',
        'exclude_columns': [
            # Add any columns you want to exclude here
            # Example: 'specific_column_name'
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
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
        'test_size': 0.2,
        'random_state': 42,
        'xgb_params': {
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
    }


# Main execution
if __name__ == "__main__":
    
    print("🚀 Starting XGBoost Risk Prediction Model - Step 1")
    print("="*60)
    
    # Get configuration
    config = get_step1_config()
    
    # Create and run baseline model
    baseline_model = BaselineRiskModel(config)
    baseline_model.run_step1_pipeline()
    
    print("\n🏁 Step 1 execution completed!")
    print("Ready to proceed to Step 2: Basic Evaluation Framework")