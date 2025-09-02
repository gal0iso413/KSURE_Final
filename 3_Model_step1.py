"""
XGBoost Risk Prediction Model - Step 1: Classification Baseline Model
===================================================================

Step 1 Implementation (CLASSIFICATION VERSION):
- Load data from 1_Dataset.py output
- Basic data exploration (shape, missing values, target distribution)
- Create 4 separate XGBoost CLASSIFICATION models with default parameters
- Use simple train/test split (80/20) without considering dates
- Goal: Establish working pipeline and baseline performance with CLASSIFICATION approach

Design Decisions:
- 4 Separate Models: One for each risk_year (1,2,3,4)
- CLASSIFICATION: Direct class prediction (0,1,2,3) instead of regression + rounding
- Native Missing Handling: Let XGBoost handle missing X variables
- Target-Specific Filtering: Each model uses only rows with valid data for its specific target
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score, classification_report, accuracy_score, precision_score, recall_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
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

# Set style for better plots (apply style first so font overrides stick)
plt.style.use('default')
sns.set_palette("husl")

# Set up Korean font AFTER style and propagate to seaborn
kf = setup_korean_font()
try:
    # Ensure seaborn uses the same font
    sns.set_theme(rc={'font.family': plt.rcParams.get('font.family', ['sans-serif'])})
except Exception:
    pass


class ClassificationBaselineRiskModel:
    """
    Step 1: Classification Baseline XGBoost Model for Risk Prediction
    
    Creates 4 separate CLASSIFICATION models for predicting risk at years 1-4.
    Focuses on establishing working pipeline and baseline performance with CLASSIFICATION approach.
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
        # Nested structure: {model_name: {target_name: fitted_model}}
        self.models = {}
        # Nested predictions: {model_name: {target_name: {'classes': np.ndarray, 'actual': np.ndarray}}}
        self.predictions = {}
        self.feature_columns = None
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        # Font is set at module load; avoid duplicate setup here
        self.korean_font = None
        
        print("🚀 Classification Baseline Risk Model Initialized")
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
        X = processed_data[feature_cols]
        
        print(f"📊 Dataset structure (before target-specific filtering):")
        print(f"   • Total records: {len(processed_data):,}")
        print(f"   • Feature columns: {len(feature_cols)}")
        print(f"   • Target variables: {len([t for t in target_cols if t in processed_data.columns])}")
        
        # Show target availability statistics
        print(f"\n🎯 Target variable availability:")
        for target in target_cols:
            if target in processed_data.columns:
                non_null_count = processed_data[target].notna().sum()
                print(f"   • {target}: {non_null_count:,} records ({non_null_count/len(processed_data)*100:.1f}%)")
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Create a single shared split (indices) to keep alignment across all targets
        print(f"\n🔄 Creating train/test split ({int((1-self.config['test_size'])*100)}/{int(self.config['test_size']*100)})...")
        indices = np.arange(len(processed_data))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            shuffle=True
        )
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # Slice each target by the same indices (keep NaNs; per-target filtering later)
        y_train_dict: Dict[str, np.ndarray] = {}
        y_test_dict: Dict[str, np.ndarray] = {}
        for target in target_cols:
            if target in processed_data.columns:
                y_full = processed_data[target].values
                y_train_dict[target] = y_full[train_idx]
                y_test_dict[target] = y_full[test_idx]
        
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
        Train separate XGBoost CLASSIFICATION models for each target
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "="*60)
        print("3️⃣ MODEL TRAINING (CLASSIFICATION) - XGBoost (baseline only)")
        print("="*60)
        
        model_store: Dict[str, Dict[str, object]] = {}
        xgb_params = self.config.get('xgb_params', {})

        # XGBoost baseline
        xgb_defaults = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1,
        }
        xgb_defaults.update(xgb_params)

        # Baseline candidate: XGBoost only
        candidates = {
            'xgboost': xgb.XGBClassifier(**xgb_defaults),
        }

        print(f"🔧 Models to train: {list(candidates.keys())}")
        print(f"📈 Training separate models per target for each algorithm with target-specific filtering...")

        for model_name, estimator in candidates.items():
            print(f"\n==============================")
            print(f"🚀 Training algorithm: {model_name}")
            per_target_models: Dict[str, object] = {}
            for target_name, y_train_values in self.y_train.items():
                print(f"\n   🎯 Training model for {target_name}...")
                try:
                    valid_mask = ~pd.isna(y_train_values)
                    X_train_filtered = self.X_train[valid_mask]
                    y_train_filtered = y_train_values[valid_mask]
                    if len(y_train_filtered) == 0:
                        print(f"      ❌ No valid training data for {target_name}")
                        continue
                    y_train_filtered = y_train_filtered.astype(int)

                    print(f"      📊 Training samples: {len(y_train_filtered):,} (filtered from {len(y_train_values):,})")
                    print(f"      🎯 Target classes: {sorted(np.unique(y_train_filtered))}")

                    # Fit a fresh clone per target to avoid estimator reuse side-effects
                    # Fresh estimator per target
                    model = xgb.XGBClassifier(**xgb_defaults)

                    print(f"      🔄 Training {model_name}...")
                    model.fit(X_train_filtered, y_train_filtered)
                    per_target_models[target_name] = model

                    pred = model.predict(X_train_filtered)
                    accuracy = accuracy_score(y_train_filtered, pred)
                    f1 = f1_score(y_train_filtered, pred, average='macro')
                    print(f"      ✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                except Exception as e:
                    print(f"      ❌ Error training {model_name} for {target_name}: {e}")
                    continue
            model_store[model_name] = per_target_models

        self.models = model_store
        total_models = sum(len(m) for m in model_store.values())
        print(f"\n✅ Training completed: {total_models} models across {len(model_store)} algorithms ready")
        return model_store
    
    def predict_and_evaluate(self) -> Dict:
        """
        Generate predictions and evaluate CLASSIFICATION models
        
        Returns:
            Dictionary of predictions and metrics
        """
        print("\n" + "="*60)
        print("4️⃣ PREDICTION & EVALUATION (CLASSIFICATION) - by algorithm")
        print("="*60)
        
        all_predictions: Dict[str, Dict] = {}
        print(f"🔮 Generating predictions for algorithms: {list(self.models.keys())}")

        for model_name, per_target_models in self.models.items():
            print(f"\n==============================")
            print(f"🧪 Evaluating algorithm: {model_name}")
            predictions = {}
            metrics = {}
            for target_name, model in per_target_models.items():
                print(f"\n   📊 Evaluating {target_name}...")
                try:
                    y_test_values = self.y_test[target_name]
                    valid_mask = ~pd.isna(y_test_values)
                    X_test_filtered = self.X_test[valid_mask]
                    y_test_filtered = y_test_values[valid_mask]
                    if len(y_test_filtered) == 0:
                        print(f"      ❌ No valid test data for {target_name}")
                        continue
                    y_test_filtered = y_test_filtered.astype(int)
                    print(f"      📊 Test samples: {len(y_test_filtered):,} (filtered from {len(y_test_values):,})")

                    y_pred_classes = model.predict(X_test_filtered)
                    predictions[target_name] = {
                        'classes': y_pred_classes,
                        'actual': y_test_filtered
                    }
                    accuracy = accuracy_score(y_test_filtered, y_pred_classes)
                    f1_macro = f1_score(y_test_filtered, y_pred_classes, average='macro')
                    precision_macro = precision_score(y_test_filtered, y_pred_classes, average='macro')
                    recall_macro = recall_score(y_test_filtered, y_pred_classes, average='macro')
                    metrics[target_name] = {
                        'accuracy': accuracy,
                        'f1_macro': f1_macro,
                        'precision_macro': precision_macro,
                        'recall_macro': recall_macro,
                        'prediction_distribution': dict(pd.Series(y_pred_classes).value_counts().sort_index())
                    }
                    print(f"      📈 Accuracy: {accuracy:.4f}")
                    print(f"      🎯 F1-Score (Macro): {f1_macro:.4f}")
                    print(f"      📊 Precision (Macro): {precision_macro:.4f}")
                    print(f"      📊 Recall (Macro): {recall_macro:.4f}")
                except Exception as e:
                    print(f"      ❌ Error evaluating {model_name} for {target_name}: {e}")
                    continue

            self.predictions[model_name] = predictions

            # Print evaluation summary
            print(f"\n📋 EVALUATION SUMMARY ({model_name}):")
            if metrics:
                avg_accuracy = np.mean([m['accuracy'] for m in metrics.values()])
                avg_f1 = np.mean([m['f1_macro'] for m in metrics.values()])
                print(f"   📊 Average Accuracy: {avg_accuracy:.4f}")
                print(f"   📊 Average F1-Score: {avg_f1:.4f}")

            # Save metrics per model
            self._save_performance_metrics(model_name, self._format_metrics_for_saving(metrics))

        return self.predictions
    
    def create_visualizations(self):
        """
        Create basic visualizations for CLASSIFICATION model performance
        """
        print("\n" + "="*60)
        print("5️⃣ VISUALIZATION & SUMMARY (CLASSIFICATION - XGBoost baseline)")
        print("="*60)
        
        if not self.predictions:
            print("❌ No predictions available for visualization")
            return
        
        # For each algorithm, create plots and summaries
        for model_name, preds in self.predictions.items():
            # Create figure with subplots
            n_targets = len(preds)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Classification Baseline Performance - {model_name.upper()}', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            # 1. Prediction vs Actual scatter plots per target (up to 4)
            for i, (target_name, pred_data) in enumerate(preds.items()):
                if i >= 4:
                    break
                ax = axes[i]
                y_true = pred_data['actual']
                y_pred = pred_data['classes']
                ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                min_val, max_val = 0, 3
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                ax.set_xlabel('Actual Risk Level')
                ax.set_ylabel('Predicted Risk Level')
                ax.set_title(f'{target_name} - Predictions vs Actual')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-0.5, 3.5)
                ax.set_ylim(-0.5, 3.5)
                accuracy = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro')
                ax.text(0.05, 0.95, f'Accuracy: {accuracy:.3f}\nF1: {f1_macro:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.tight_layout()
            self._save_plot(f"{model_name}_01_classification_prediction_vs_actual")

            # 2. Feature Importance Analysis (only for models with attribute)
            self._plot_feature_importance(model_name)

            # 3. Performance summary
            self._print_performance_summary(model_name)
    
    def _plot_feature_importance(self, model_name: str):
        """Plot feature importance for models that support it (XGBoost/RandomForest)."""
        if model_name not in self.models:
            return
        per_target = self.models[model_name]
        print(f"\n📊 Feature Importance Analysis (Classification) - {model_name}")
        importance_data = {}
        for target_name, model in per_target.items():
            # For pipelines, extract final estimator
            estimator = model
            if isinstance(model, Pipeline):
                estimator = model.named_steps.get('mlp', model)
            if hasattr(estimator, 'feature_importances_'):
                importance_data[target_name] = estimator.feature_importances_
        if not importance_data:
            print("   ❌ No feature importance data available for this algorithm")
            return
        importance_df = pd.DataFrame(importance_data, index=self.feature_columns)
        importance_df['avg_importance'] = importance_df.mean(axis=1)
        top_features = importance_df.nlargest(20, 'avg_importance')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        fig.suptitle(f'Feature Importance Analysis (Classification) - {model_name.upper()}', fontsize=16, fontweight='bold')
        top_features['avg_importance'].plot(kind='barh', ax=ax1)
        ax1.set_title('Top 20 Features - Average Importance')
        ax1.set_xlabel('Importance Score')
        target_cols = [col for col in top_features.columns if col != 'avg_importance']
        if target_cols:
            plot_data = top_features[target_cols].T
            plot_data.plot(kind='barh', ax=ax2)
            ax2.set_title('Feature Importance by Target Variable')
            ax2.set_xlabel('Importance Score')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.85, top=0.9)
        self._save_plot(f"{model_name}_02_classification_feature_importance")
        print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES ({model_name}):")
        for i, (feature, importance) in enumerate(top_features['avg_importance'].head(10).items(), 1):
            print(f"   {i:2d}. {feature}: {importance:.4f}")
    
    def _print_performance_summary(self, model_name: str):
        """Print comprehensive summary for a specific algorithm using cached predictions."""
        if model_name not in self.predictions:
            return
        print(f"\n📊 COMPREHENSIVE CLASSIFICATION PERFORMANCE SUMMARY - {model_name}")
        print("-" * 60)
        preds = self.predictions[model_name]
        summary_data = []
        for target_name, pred_data in preds.items():
            y_true = pred_data['actual']
            y_pred = pred_data['classes']
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')
            precision_macro = precision_score(y_true, y_pred, average='macro')
            recall_macro = recall_score(y_true, y_pred, average='macro')
            true_dist = pd.Series(y_true).value_counts().sort_index()
            pred_dist = pd.Series(y_pred).value_counts().sort_index()
            summary_data.append({
                'Target': target_name,
                'Accuracy': f"{accuracy:.4f}",
                'F1_Macro': f"{f1_macro:.4f}",
                'Precision_Macro': f"{precision_macro:.4f}",
                'Recall_Macro': f"{recall_macro:.4f}",
                'True_Dist': dict(true_dist),
                'Pred_Dist': dict(pred_dist)
            })
        for data in summary_data:
            print(f"\n🎯 {data['Target']}:")
            print(f"   • Accuracy: {data['Accuracy']}")
            print(f"   • F1-Score (Macro): {data['F1_Macro']}")
            print(f"   • Precision (Macro): {data['Precision_Macro']}")
            print(f"   • Recall (Macro): {data['Recall_Macro']}")
            print(f"   • Actual distribution: {data['True_Dist']}")
            print(f"   • Predicted distribution: {data['Pred_Dist']}")
    
    def _save_performance_metrics(self, model_name: str, summary_data: List[Dict]):
        """
        Save CLASSIFICATION performance metrics to CSV and JSON files for a specific algorithm
        
        Args:
            model_name: Algorithm name
            summary_data: List of performance data dictionaries (already flattened)
        """
        print("\n💾 Saving CLASSIFICATION performance metrics to files...")
        
        # Nothing to save (e.g., model failed to train for all targets)
        if not summary_data:
            print("  ⚠️ No metrics to save for this algorithm; skipping.")
            return
        
        # Prepare data for CSV (flatten dictionaries)
        csv_data = summary_data
        
        # Save CSV
        csv_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.results_dir, 'metrics', f'step1_performance_summary_{model_name}.csv')
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  📊 Classification performance summary saved: step1_performance_summary_{model_name}.csv")
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, 'metrics', f'step1_detailed_results_{model_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary_data, f, indent=2, default=str, ensure_ascii=False)
        print(f"  📄 Classification detailed results saved: step1_detailed_results_{model_name}.json")
        
        # Save model files
        self._save_models(model_name)

    def _format_metrics_for_saving(self, metrics: Dict[str, Dict]) -> List[Dict]:
        """
        Flatten per-target metrics dict into a list of rows suitable for CSV/JSON saving.
        Includes an additional 'AVERAGE' row if metrics exist.
        """
        formatted_rows: List[Dict] = []
        for target_name, metric_values in metrics.items():
            formatted_rows.append({
                'target': target_name,
                'accuracy': float(metric_values.get('accuracy', np.nan)),
                'f1_macro': float(metric_values.get('f1_macro', np.nan)),
                'precision_macro': float(metric_values.get('precision_macro', np.nan)),
                'recall_macro': float(metric_values.get('recall_macro', np.nan)),
                'prediction_distribution': metric_values.get('prediction_distribution', {})
            })
        if formatted_rows:
            avg_row = {
                'target': 'AVERAGE',
                'accuracy': float(np.nanmean([row['accuracy'] for row in formatted_rows])),
                'f1_macro': float(np.nanmean([row['f1_macro'] for row in formatted_rows])),
                'precision_macro': float(np.nanmean([row['precision_macro'] for row in formatted_rows])),
                'recall_macro': float(np.nanmean([row['recall_macro'] for row in formatted_rows])),
                'prediction_distribution': {}
            }
            formatted_rows.append(avg_row)
        return formatted_rows
    
    def _save_models(self, model_name: str):
        """Save trained CLASSIFICATION models to files for a specific algorithm"""
        print("\n💾 Saving trained CLASSIFICATION models...")
        if model_name not in self.models:
            return
        for target_name, model in self.models[model_name].items():
            if model_name == 'xgboost':
                model_path = os.path.join(self.results_dir, 'models', f'{target_name}_{model_name}_model.json')
                model.save_model(model_path)
            else:
                model_path = os.path.join(self.results_dir, 'models', f'{target_name}_{model_name}_model.joblib')
                joblib.dump(model, model_path)
            print(f"  🤖 Classification model saved: {os.path.basename(model_path)}")
    
    def run_step1_classification_pipeline(self):
        """
        Execute complete Step 1 CLASSIFICATION pipeline
        """
        print("🏗️ EXECUTING STEP 1: CLASSIFICATION BASELINE MODEL")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Train models for all three algorithms
            self.train_models()

            # Step 4: Predict and evaluate per algorithm
            self.predict_and_evaluate()

            # Step 5: Create per-algorithm visualizations and summaries
            self.create_visualizations()
            
            print("\n🎉 STEP 1 CLASSIFICATION COMPLETED SUCCESSFULLY!")
            print("✅ Classification baseline performance established")
            print("✅ Working classification pipeline ready")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 1 CLASSIFICATION FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 CLASSIFICATION RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Visualizations:")
        print(f"      • xgboost_01_classification_prediction_vs_actual.png")
        print(f"      • xgboost_02_classification_feature_importance.png")
        print(f"   📈 Metrics:")
        print(f"      • step1_performance_summary_xgboost.csv / step1_detailed_results_xgboost.json")
        print(f"   🤖 Models:")
        for target in self.config['target_columns']:
            print(f"      • {target}_xgboost_model.json")
        print(f"\n💡 Use these files for comparison with other steps!")


def get_step1_classification_config():
    """
    Configuration for Step 1 CLASSIFICATION baseline model
    """
    return {
        'data_path': 'dataset/credit_risk_dataset.csv',
        'exclude_columns': [
            # Add any columns you want to exclude here
            '사업자등록번호',
            '대상자명',
            '청약번호',
            '보험청약일자',
            '수출자대상자번호',
            '업종코드1'
        ],
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
        'test_size': 0.2,
        'random_state': 42,
        'xgb_params': {
            'objective': 'multi:softprob',
            'num_class': 4,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'eval_metric': 'mlogloss',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1
        }
    }


# Main execution
if __name__ == "__main__":
    
    print("🚀 Starting XGBoost Risk Prediction Model - Step 1 (CLASSIFICATION)")
    print("="*60)
    
    # Get configuration
    config = get_step1_classification_config()
    
    # Create and run classification baseline model
    classification_baseline_model = ClassificationBaselineRiskModel(config)
    classification_baseline_model.run_step1_classification_pipeline()
    
    print("\n🏁 Step 1 Classification execution completed!")
    print("Ready to proceed to Step 2!")
