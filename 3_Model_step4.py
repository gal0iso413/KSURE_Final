"""
XGBoost Risk Prediction Model - Step 4: Temporal Train/Test Split
============================================================

Step 4 Implementation:
- Load data from 1_Dataset.py output
- Implement temporal train/test split using 보험청약일자
- Use chronological 80/20 split (no shuffling)
- Train 4 separate XGBoost models with temporal awareness
- Goal: Prevent temporal leakage and ensure realistic evaluation

Design Decisions:
- Temporal Split: Sort by 보험청약일자, use first 80% for training, last 20% for testing
- 4 Separate Models: One for each risk_year (1,2,3,4)
- Regression with Rounding: Treat as continuous then round to preserve ordinality
- Native Missing Handling: Let XGBoost handle missing X variables
- Target-Specific Filtering: Each model uses only rows with valid data for its specific target
- Simple Date-Based Split: Use all data chronologically, let XGBoost handle missing targets naturally
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings
import os
import platform
import matplotlib.font_manager as fm
import json
from datetime import datetime
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


class TemporalSplitModel:
    """
    Step 4: Temporal Train/Test Split XGBoost Model for Risk Prediction
    
    Implements temporal-aware train/test split using 보험청약일자 to prevent
    temporal leakage and ensure realistic evaluation of future predictions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize temporal split model with configuration
        
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
        self.results_dir = None
        self.split_date = None
        self.temporal_stats = {}
        
        print("🔍 Temporal Split Model Initialized")
        print(f"📅 Temporal reference: {config.get('temporal_column', '보험청약일자')}")
        print(f"📊 Split ratio: {config.get('train_ratio', 0.8):.1%} train / {1-config.get('train_ratio', 0.8):.1%} test")
    
    def _create_results_directory(self) -> str:
        """Create results directory for Step 4 outputs"""
        results_dir = "result/step4_temporal_split"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'temporal_analysis'), exist_ok=True)
        
        self.results_dir = results_dir
        print(f"📁 Results will be saved to: {results_dir}")
        return results_dir
    
    def _save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
        """Save plot with Korean font support"""
        plot_path = os.path.join(self.results_dir, 'visualizations', f"{filename}.png")
        plt.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"  💾 Saved: {filename}.png")
        return plot_path
    
    def load_and_explore_data(self) -> pd.DataFrame:
        """
        Load data and perform basic exploration with temporal focus
        """
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING & TEMPORAL EXPLORATION")
        print("="*60)
        
        try:
            # Load data
            self.data = pd.read_csv(self.config['data_path'], encoding='utf-8')
            print(f"✅ Successfully loaded {self.config['data_path']} with utf-8 encoding")
            print(f"✅ Data loaded: {self.data.shape}")
            
            # Basic dataset structure
            print(f"\n📊 Dataset structure:")
            print(f"   • Total records: {len(self.data):,}")
            print(f"   • Total features: {len(self.data.columns)}")
            print(f"   • Target variables: {len(self.config['target_columns'])}")
            
            # Temporal column analysis
            temporal_col = self.config['temporal_column']
            if temporal_col in self.data.columns:
                self.data[temporal_col] = pd.to_datetime(self.data[temporal_col], format='%Y-%m-%d')
                print(f"\n📅 Temporal analysis:")
                print(f"   • Temporal column: {temporal_col}")
                print(f"   • Date range: {self.data[temporal_col].min()} to {self.data[temporal_col].max()}")
                print(f"   • Total days: {(self.data[temporal_col].max() - self.data[temporal_col].min()).days}")
                
                # Check for missing temporal data
                missing_temporal = self.data[temporal_col].isna().sum()
                if missing_temporal > 0:
                    print(f"   ⚠️  Missing temporal data: {missing_temporal} records")
                else:
                    print(f"   ✅ No missing temporal data")
            else:
                raise ValueError(f"Temporal column '{temporal_col}' not found in dataset")
            
            return self.data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def implement_temporal_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Implement simple temporal train/test split using chronological ordering
        
        Returns:
            Tuple of (X_train, X_test, y_train_dict, y_test_dict)
        """
        print("\n" + "="*60)
        print("2️⃣ TEMPORAL TRAIN/TEST SPLIT")
        print("="*60)
        
        # Sort data chronologically by temporal column
        temporal_col = self.config['temporal_column']
        self.data = self.data.sort_values(temporal_col).reset_index(drop=True)
        
        # Simple chronological split
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        print(f"📊 Simple temporal split details:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Train records: {train_size:,} ({train_size/len(self.data):.1%})")
        print(f"   • Test records: {len(self.data) - train_size:,} ({(len(self.data) - train_size)/len(self.data):.1%})")
        
        # Get the actual split date for reporting
        self.split_date = self.data.iloc[train_size-1][temporal_col]
        print(f"   • Split date: {self.split_date}")
        
        # Split data
        train_data = self.data.iloc[:train_size].copy()
        test_data = self.data.iloc[train_size:].copy()
        
        # Separate features and targets
        exclude_cols = self.config.get('exclude_columns', [])
        target_cols = self.config['target_columns']
        
        # Feature columns (exclude targets, excluded columns, and temporal column)
        feature_cols = [col for col in self.data.columns 
                       if col not in target_cols and col not in exclude_cols and col != temporal_col]
        
        print(f"\n📊 Feature analysis:")
        print(f"   • Total features: {len(feature_cols)}")
        print(f"   • Target variables: {len(target_cols)}")
        print(f"   • Excluded columns: {len(exclude_cols)}")
        print(f"   • Temporal column excluded from features: {temporal_col}")
        
        # Prepare X and y data
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        
        y_train = {}
        y_test = {}
        
        # Prepare target variables
        print(f"\n🎯 Target variable analysis:")
        for target in target_cols:
            # Train set
            y_train[target] = train_data[target]
            train_available = y_train[target].notna().sum()
            train_total = len(y_train[target])
            
            # Test set
            y_test[target] = test_data[target]
            test_available = y_test[target].notna().sum()
            test_total = len(y_test[target])
            
            print(f"   • {target}:")
            print(f"     - Train: {train_available:,}/{train_total:,} available ({train_available/train_total:.1%})")
            print(f"     - Test: {test_available:,}/{test_total:,} available ({test_available/test_total:.1%})")
            
            # Store temporal statistics
            self.temporal_stats[target] = {
                'train_total': train_total,
                'train_available': train_available,
                'train_missing': train_total - train_available,
                'test_total': test_total,
                'test_available': test_available,
                'test_missing': test_total - test_available
            }
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self) -> Dict:
        """
        Train separate XGBoost regression models for each target with temporal awareness
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("3️⃣ MODEL TRAINING WITH TEMPORAL AWARENESS")
        print("="*60)
        
        models = {}
        xgb_params = self.config.get('xgb_params', {})
        
        # Default parameters for temporal model
        default_params = {
            'objective': 'reg:squarederror',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
        default_params.update(xgb_params)
        
        print(f"🔧 XGBoost parameters: {default_params}")
        print(f"📈 Training {len(self.y_train)} separate models with temporal filtering...")
        
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
        Generate predictions and evaluate model performance with temporal awareness
        
        Returns:
            Dictionary of predictions and metrics
        """
        print("\n" + "="*60)
        print("4️⃣ PREDICTION & TEMPORAL EVALUATION")
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
                
                # Round predictions to nearest integer for classification
                y_pred_rounded = np.round(y_pred_raw).astype(int)
                
                # Clip predictions to valid range [0, 3]
                y_pred_clipped = np.clip(y_pred_rounded, 0, 3)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test_filtered, y_pred_raw)
                accuracy = accuracy_score(y_test_filtered, y_pred_clipped)
                
                # Classification report
                class_report = classification_report(y_test_filtered, y_pred_clipped, 
                                                  output_dict=True, zero_division=0)
                
                # Store results
                predictions[target_name] = {
                    'y_true': y_test_filtered,
                    'y_pred_raw': y_pred_raw,
                    'y_pred_rounded': y_pred_clipped
                }
                
                metrics[target_name] = {
                    'mae': mae,
                    'accuracy': accuracy,
                    'classification_report': class_report,
                    'sample_size': len(y_test_filtered)
                }
                
                print(f"      ✅ Evaluation completed")
                print(f"      📊 MAE: {mae:.4f}")
                print(f"      📊 Accuracy: {accuracy:.4f}")
                print(f"      📊 Sample size: {len(y_test_filtered):,}")
                
            except Exception as e:
                print(f"      ❌ Error evaluating {target_name}: {e}")
                continue
        
        self.predictions = predictions
        self.metrics = metrics
        
        return predictions, metrics
    
    def create_visualizations(self):
        """Create comprehensive visualizations for temporal split analysis"""
        print("\n" + "="*60)
        print("5️⃣ TEMPORAL VISUALIZATIONS")
        print("="*60)
        
        # 1. Temporal split visualization
        self._plot_temporal_split()
        
        # 2. Target distribution comparison
        self._plot_target_distributions()
        
        # 3. Performance metrics visualization
        self._plot_performance_metrics()
        
        # 4. Temporal drift analysis
        self._plot_temporal_drift()
    
    def _plot_temporal_split(self):
        """Visualize the temporal split"""
        temporal_col = self.config['temporal_column']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate split indices
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        # Plot 1: Date distribution
        train_dates = self.data.iloc[:train_size][temporal_col]
        test_dates = self.data.iloc[train_size:][temporal_col]
        
        ax1.hist(train_dates, bins=50, alpha=0.7, label='Train Set', color='blue')
        ax1.hist(test_dates, bins=50, alpha=0.7, label='Test Set', color='red')
        ax1.axvline(self.split_date, color='black', linestyle='--', label=f'Split Date: {self.split_date}')
        ax1.set_title('Temporal Train/Test Split Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Contracts')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Monthly distribution
        train_monthly = train_dates.dt.to_period('M').value_counts().sort_index()
        test_monthly = test_dates.dt.to_period('M').value_counts().sort_index()
        
        ax2.plot(train_monthly.index.astype(str), train_monthly.values, 
                marker='o', label='Train Set', color='blue')
        ax2.plot(test_monthly.index.astype(str), test_monthly.values, 
                marker='s', label='Test Set', color='red')
        ax2.set_title('Monthly Contract Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Contracts')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self._save_plot("01_temporal_split")
    
    def _plot_target_distributions(self):
        """Compare target distributions between train and test sets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Variable Distributions: Train vs Test', fontsize=16, fontweight='bold')
        
        for i, target in enumerate(self.config['target_columns']):
            ax = axes[i//2, i%2]
            
            if target in self.y_train and target in self.y_test:
                # Get available data
                train_data = self.y_train[target].dropna()
                test_data = self.y_test[target].dropna()
                
                if len(train_data) > 0 and len(test_data) > 0:
                    # Get all unique risk levels from both sets
                    all_levels = sorted(set(train_data.unique()) | set(test_data.unique()))
                    
                    # Count occurrences for each level, filling with 0 if missing
                    train_counts = train_data.value_counts().reindex(all_levels, fill_value=0)
                    test_counts = test_data.value_counts().reindex(all_levels, fill_value=0)
                    
                    x = np.arange(len(all_levels))
                    width = 0.35
                    
                    ax.bar(x - width/2, train_counts.values, width, label='Train', alpha=0.7)
                    ax.bar(x + width/2, test_counts.values, width, label='Test', alpha=0.7)
                    
                    ax.set_title(f'{target} Distribution', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Risk Level')
                    ax.set_ylabel('Count')
                    ax.set_xticks(x)
                    ax.set_xticklabels(all_levels)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("02_target_distributions")
    
    def _plot_performance_metrics(self):
        """Visualize performance metrics across targets"""
        if not hasattr(self, 'metrics') or not self.metrics:
            print("   ⚠️  No metrics available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract metrics
        targets = list(self.metrics.keys())
        mae_scores = [self.metrics[t]['mae'] for t in targets]
        accuracy_scores = [self.metrics[t]['accuracy'] for t in targets]
        sample_sizes = [self.metrics[t]['sample_size'] for t in targets]
        
        # Plot 1: MAE and Accuracy
        x = np.arange(len(targets))
        width = 0.35
        
        ax1.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.7, color='red')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, accuracy_scores, width, label='Accuracy', alpha=0.7, color='green')
        
        ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Target Variable')
        ax1.set_ylabel('MAE', color='red')
        ax1_twin.set_ylabel('Accuracy', color='green')
        ax1.set_xticks(x)
        ax1.set_xticklabels(targets)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample sizes
        ax2.bar(targets, sample_sizes, alpha=0.7, color='blue')
        ax2.set_title('Test Set Sample Sizes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Target Variable')
        ax2.set_ylabel('Sample Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("03_performance_metrics")
    
    def _plot_temporal_drift(self):
        """Analyze temporal drift within test set"""
        temporal_col = self.config['temporal_column']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Drift Analysis in Test Set', fontsize=16, fontweight='bold')
        
        # Calculate split indices
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        test_data = self.data.iloc[train_size:].copy()
        
        for i, target in enumerate(self.config['target_columns'][:4]):  # Limit to 4 plots
            ax = axes[i//2, i%2]
            
            if target in test_data.columns:
                # Group by month and calculate average risk
                test_data['month'] = test_data[temporal_col].dt.to_period('M')
                monthly_risk = test_data.groupby('month')[target].agg(['mean', 'count']).reset_index()
                
                if len(monthly_risk) > 1:
                    ax.plot(monthly_risk['month'].astype(str), monthly_risk['mean'], 
                           marker='o', linewidth=2, markersize=6)
                    ax.set_title(f'{target} - Monthly Average Risk', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Month')
                    ax.set_ylabel('Average Risk Level')
                    ax.grid(True, alpha=0.3)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        self._save_plot("04_temporal_drift")
    
    def _print_performance_summary(self):
        """Print comprehensive performance summary"""
        print("\n" + "="*60)
        print("📊 TEMPORAL SPLIT PERFORMANCE SUMMARY")
        print("="*60)
        
        # Calculate split statistics
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        print(f"📅 Split Information:")
        print(f"   • Split date: {self.split_date}")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Train records: {train_size:,} ({train_size/len(self.data):.1%})")
        print(f"   • Test records: {len(self.data) - train_size:,} ({(len(self.data) - train_size)/len(self.data):.1%})")
        print(f"   • Temporal column: {self.config['temporal_column']} (excluded from features)")
        
        print(f"\n🎯 Target Variable Coverage:")
        for target, stats in self.temporal_stats.items():
            print(f"   • {target}:")
            print(f"     - Train: {stats['train_available']:,}/{stats['train_total']:,} ({stats['train_available']/stats['train_total']:.1%})")
            print(f"     - Test: {stats['test_available']:,}/{stats['test_total']:,} ({stats['test_available']/stats['test_total']:.1%})")
        
        if hasattr(self, 'metrics'):
            print(f"\n📈 Model Performance:")
            for target, metric in self.metrics.items():
                print(f"   • {target}:")
                print(f"     - MAE: {metric['mae']:.4f}")
                print(f"     - Accuracy: {metric['accuracy']:.4f}")
                print(f"     - Test samples: {metric['sample_size']:,}")
    
    def _save_performance_metrics(self, summary_data: List[Dict]):
        """Save detailed performance metrics"""
        # Save comprehensive metrics
        metrics_file = os.path.join(self.results_dir, 'metrics', 'step4_detailed_results.json')
        
        # Calculate split statistics
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        results_summary = {
            'split_info': {
                'split_method': 'simple_temporal',
                'split_date': str(self.split_date),
                'train_ratio': self.config.get('train_ratio', 0.8),
                'temporal_column': self.config['temporal_column'],
                'total_records': len(self.data),
                'train_records': train_size,
                'test_records': len(self.data) - train_size
            },
            'temporal_stats': self.temporal_stats,
            'model_metrics': self.metrics if hasattr(self, 'metrics') else {},
            'summary_data': summary_data
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  💾 Saved detailed metrics: step4_detailed_results.json")
    
    def _save_models(self):
        """Save trained models"""
        print(f"\n🤖 Saving trained models...")
        
        for target_name, model in self.models.items():
            model_path = os.path.join(self.results_dir, 'models', f'{target_name}_model.json')
            model.save_model(model_path)
            print(f"  🤖 Model saved: {target_name}_model.json")
    
    def run_step4_pipeline(self):
        """
        Execute complete Step 4 pipeline with temporal split
        """
        print("🏗️ EXECUTING STEP 4: TEMPORAL TRAIN/TEST SPLIT")
        print("=" * 70)
        
        try:
            # Create results directory first
            self._create_results_directory()
            
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Implement temporal split
            self.implement_temporal_split()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Predict and evaluate
            self.predict_and_evaluate()
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            # Step 6: Save results
            self._save_performance_metrics([])
            self._save_models()
            
            print("\n🎉 STEP 4 COMPLETED SUCCESSFULLY!")
            print("✅ Temporal split implemented")
            print("✅ No temporal leakage ensured")
            print("✅ Realistic evaluation achieved")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 4 FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Visualizations:")
        print(f"      • 01_temporal_split.png")
        print(f"      • 02_target_distributions.png")
        print(f"      • 03_performance_metrics.png")
        print(f"      • 04_temporal_drift.png")
        print(f"   📈 Metrics:")
        print(f"      • step4_detailed_results.json")
        print(f"   🤖 Models:")
        for target in self.config['target_columns']:
            print(f"      • {target}_model.json")
        print(f"\n💡 Use these files for offline analysis and reporting!")


def get_step4_config():
    """
    Configuration for Step 4 temporal split model
    """
    return {
        'data_path': 'dataset/credit_risk_dataset.csv',
        'temporal_column': '보험청약일자',
        'train_ratio': 0.8,
        'exclude_columns': [
            '사업자등록번호',
            '대상자명',
            '대상자등록이력일시',
            '대상자기본주소',
            '청약번호',
            '청약상태코드',
            '수출자대상자번호',
            '특별출연협약코드',
            '업종코드1'
        ],
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
        'random_state': 42,
        'xgb_params': {
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
    }


# Main execution
if __name__ == "__main__":
    
    print("🚀 Starting XGBoost Risk Prediction Model - Step 4")
    print("="*60)
    
    # Get configuration
    config = get_step4_config()
    
    # Create and run temporal split model
    temporal_model = TemporalSplitModel(config)
    temporal_model.run_step4_pipeline()
    
    print("\n🏁 Step 4 execution completed!")
    print("Ready to proceed to Step 5: Class Imbalance Strategy") 