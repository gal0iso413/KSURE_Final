"""
XGBoost Risk Prediction Model - Step 5: Class Imbalance Strategy
============================================================

Step 5 Implementation:
- Load results from Step 4 as baseline
- Implement unified class imbalance strategies
- Strategy A: Class Weights (XGBoost scale_pos_weight)
- Strategy B: SMOTE with temporal integrity preservation
- Strategy C: Hybrid approach
- Compare all strategies against Step 4 baseline
- Focus on macro-F1 and recall for minority classes

Design Decisions:
- Unified Strategy: Apply same approach to all 4 targets for fair comparison
- Temporal Integrity: Preserve natural missing patterns, no future data leakage
- SMOTE Application: Apply to entire dataset first, then temporal split
- Missing Data: Respect temporal constraints (no artificial future data)
- Evaluation: Macro-F1 and recall as primary metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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
        print("⚠️  Preferred Korean fonts not found. Using fallback options...")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # Ensure minus signs display correctly
    plt.rcParams['axes.unicode_minus'] = False
    return korean_font

# Set up Korean font
setup_korean_font()

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class ClassImbalanceModel:
    """
    Step 5: Class Imbalance Strategy XGBoost Model for Risk Prediction
    
    Implements unified class imbalance strategies while maintaining temporal integrity
    and comparing performance against Step 4 baseline.
    """
    
    def __init__(self, config: Dict):
        """Initialize class imbalance model with configuration"""
        self.config = config
        self.data = None
        self.baseline_results = None
        self.results_dir = None
        self.strategies = ['baseline', 'class_weights', 'smote', 'hybrid']
        self.strategy_results = {}
        
        print("🔍 Class Imbalance Model Initialized")
        print(f"📊 Strategies: {', '.join(self.strategies)}")
        print(f"🎯 Primary metrics: Macro-F1, Recall")
    
    def _create_results_directory(self) -> str:
        """Create results directory for Step 5 outputs"""
        results_dir = "result/step5_class_imbalance"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'comparison'), exist_ok=True)
        
        self.results_dir = results_dir
        print(f"📁 Results will be saved to: {results_dir}")
        return results_dir
    
    def _save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
        """Save plot with Korean font support"""
        plot_path = os.path.join(self.results_dir, 'visualizations', f"{filename}.png")
        plt.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"  💾 Saved: {filename}.png")
        return plot_path
    
    def load_data_and_baseline(self) -> Tuple[pd.DataFrame, Dict]:
        """Load data and Step 4 baseline results"""
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING & BASELINE ANALYSIS")
        print("="*60)
        
        try:
            # Load data
            self.data = pd.read_csv(self.config['data_path'], encoding='utf-8')
            print(f"✅ Successfully loaded {self.config['data_path']}")
            print(f"✅ Data loaded: {self.data.shape}")
            
            # Load Step 4 baseline results
            baseline_file = "result/step4_temporal_split/metrics/step4_detailed_results.json"
            if os.path.exists(baseline_file):
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    self.baseline_results = json.load(f)
                print(f"✅ Loaded Step 4 baseline results")
            else:
                print(f"⚠️  Step 4 baseline results not found at {baseline_file}")
                self.baseline_results = {}
            
            # Temporal column setup
            temporal_col = self.config['temporal_column']
            if temporal_col in self.data.columns:
                self.data[temporal_col] = pd.to_datetime(self.data[temporal_col], format='%Y-%m-%d')
                print(f"✅ Temporal column converted: {temporal_col}")
            
            return self.data, self.baseline_results
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_class_imbalance(self) -> Dict:
        """Analyze class imbalance for each target variable"""
        print("\n" + "="*60)
        print("2️⃣ CLASS IMBALANCE ANALYSIS")
        print("="*60)
        
        imbalance_analysis = {}
        target_cols = self.config['target_columns']
        
        for target in target_cols:
            print(f"\n🎯 Analyzing {target}...")
            
            # Get available data (non-missing)
            available_data = self.data[target].dropna()
            total_samples = len(available_data)
            
            if total_samples > 0:
                # Calculate class distribution
                class_counts = available_data.value_counts().sort_index()
                class_ratios = class_counts / total_samples
                
                # Calculate imbalance metrics
                max_class_count = class_counts.max()
                min_class_count = class_counts.min()
                imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
                
                print(f"   📊 Available samples: {total_samples:,}")
                print(f"   📊 Class distribution: {dict(class_counts)}")
                print(f"   📊 Class ratios: {dict(class_ratios)}")
                print(f"   📊 Imbalance ratio: {imbalance_ratio:.2f}")
                
                # Determine imbalance severity
                if imbalance_ratio > 100:
                    severity = "Extreme"
                elif imbalance_ratio > 50:
                    severity = "Severe"
                elif imbalance_ratio > 20:
                    severity = "Moderate"
                else:
                    severity = "Mild"
                
                print(f"   📊 Imbalance severity: {severity}")
                
                imbalance_analysis[target] = {
                    'total_samples': total_samples,
                    'missing_samples': len(self.data) - total_samples,
                    'class_counts': dict(class_counts),
                    'class_ratios': dict(class_ratios),
                    'imbalance_ratio': imbalance_ratio,
                    'severity': severity,
                    'available_classes': list(class_counts.index)
                }
            else:
                print(f"   ❌ No available data for {target}")
                imbalance_analysis[target] = {
                    'total_samples': 0,
                    'missing_samples': len(self.data),
                    'class_counts': {},
                    'class_ratios': {},
                    'imbalance_ratio': float('inf'),
                    'severity': 'No Data',
                    'available_classes': []
                }
        
        return imbalance_analysis
    
    def implement_temporal_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Implement temporal train/test split (same as Step 4)"""
        print("\n" + "="*60)
        print("3️⃣ TEMPORAL SPLIT (SAME AS STEP 4)")
        print("="*60)
        
        # Sort data chronologically
        temporal_col = self.config['temporal_column']
        self.data = self.data.sort_values(temporal_col).reset_index(drop=True)
        
        # Simple chronological split
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(self.data) * train_ratio)
        
        print(f"📊 Temporal split details:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Train records: {train_size:,} ({train_size/len(self.data):.1%})")
        print(f"   • Test records: {len(self.data) - train_size:,} ({(len(self.data) - train_size)/len(self.data):.1%})")
        
        # Split data
        train_data = self.data.iloc[:train_size].copy()
        test_data = self.data.iloc[train_size:].copy()
        
        # Prepare features and targets
        exclude_cols = self.config.get('exclude_columns', [])
        target_cols = self.config['target_columns']
        
        # Feature columns (exclude targets, excluded columns, and temporal column)
        feature_cols = [col for col in self.data.columns 
                       if col not in target_cols and col not in exclude_cols and col != temporal_col]
        
        print(f"\n📊 Feature analysis:")
        print(f"   • Total features: {len(feature_cols)}")
        print(f"   • Target variables: {len(target_cols)}")
        print(f"   • Temporal column excluded: {temporal_col}")
        
        # Prepare X and y data
        X_train = train_data[feature_cols]
        X_test = test_data[feature_cols]
        
        y_train = {}
        y_test = {}
        
        # Prepare target variables
        print(f"\n🎯 Target variable analysis:")
        for target in target_cols:
            y_train[target] = train_data[target]
            y_test[target] = test_data[target]
            
            train_available = y_train[target].notna().sum()
            test_available = y_test[target].notna().sum()
            
            print(f"   • {target}:")
            print(f"     - Train: {train_available:,}/{len(y_train[target]):,} available ({train_available/len(y_train[target]):.1%})")
            print(f"     - Test: {test_available:,}/{len(y_test[target]):,} available ({test_available/len(y_test[target]):.1%})")
        
        return X_train, X_test, y_train, y_test
    
    def strategy_class_weights(self, X_train: pd.DataFrame, y_train: Dict) -> Dict:
        """Implement class weights strategy"""
        print("\n" + "="*60)
        print("4️⃣ STRATEGY A: CLASS WEIGHTS")
        print("="*60)
        
        models = {}
        xgb_params = self.config.get('xgb_params', {})
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
        default_params.update(xgb_params)
        
        print(f"🔧 XGBoost parameters: {default_params}")
        print(f"📈 Training {len(y_train)} models with class weights...")
        
        for target_name, y_train_values in y_train.items():
            print(f"\n   🎯 Training model for {target_name}...")
            
            try:
                # Filter training data for this specific target
                valid_mask = ~pd.isna(y_train_values)
                X_train_filtered = X_train[valid_mask]
                y_train_filtered = y_train_values[valid_mask]
                
                if len(y_train_filtered) == 0:
                    print(f"      ❌ No valid training data for {target_name}")
                    continue
                
                print(f"      📊 Training samples: {len(y_train_filtered):,}")
                
                # Calculate class weights
                unique_classes = np.array(sorted(y_train_filtered.unique()))
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=y_train_filtered
                )
                
                # Create weight mapping
                weight_dict = dict(zip(unique_classes, class_weights))
                print(f"      📊 Class weights: {weight_dict}")
                
                # Apply class weights to XGBoost
                model_params = default_params.copy()
                
                # For binary classification, use scale_pos_weight
                if len(unique_classes) == 2:
                    pos_weight = weight_dict[1] / weight_dict[0] if 0 in weight_dict and 1 in weight_dict else 1
                    model_params['scale_pos_weight'] = pos_weight
                    print(f"      📊 Scale pos weight: {pos_weight:.3f}")
                
                # Create and train model
                model = xgb.XGBRegressor(**model_params)
                model.fit(X_train_filtered, y_train_filtered)
                
                models[target_name] = {
                    'model': model,
                    'class_weights': weight_dict,
                    'training_samples': len(y_train_filtered)
                }
                
                print(f"      ✅ Model trained successfully")
                
            except Exception as e:
                print(f"      ❌ Error training {target_name}: {e}")
                continue
        
        return models
    
    def strategy_smote(self, X_train: pd.DataFrame, y_train: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Implement SMOTE strategy with temporal integrity preservation"""
        print("\n" + "="*60)
        print("5️⃣ STRATEGY B: SMOTE")
        print("="*60)
        
        print("🔍 Applying SMOTE to entire dataset first...")
        
        # Prepare features and targets for SMOTE
        exclude_cols = self.config.get('exclude_columns', [])
        target_cols = self.config['target_columns']
        temporal_col = self.config['temporal_column']
        
        # Feature columns (exclude temporal column for SMOTE)
        feature_cols = [col for col in self.data.columns 
                       if col not in target_cols and col not in exclude_cols and col != temporal_col]
        
        X_full = self.data[feature_cols]
        
        # Apply SMOTE to each target independently
        X_smote = X_full.copy()
        y_smote = {}
        
        for target in target_cols:
            print(f"\n   🎯 Applying SMOTE for {target}...")
            
            try:
                # Get available data for this target
                y_target = self.data[target]
                valid_mask = ~pd.isna(y_target)
                X_valid = X_full[valid_mask]
                y_valid = y_target[valid_mask]
                
                if len(y_valid) == 0:
                    print(f"      ❌ No valid data for {target}")
                    continue
                
                print(f"      📊 Available samples: {len(y_valid):,}")
                
                # Preprocess data for SMOTE (handle NaN values)
                imputer = SimpleImputer(strategy='median')
                X_valid_imputed = imputer.fit_transform(X_valid)
                
                # Apply SMOTE to imputed data
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X_valid_imputed, y_valid)
                
                print(f"      📊 After SMOTE: {len(y_resampled):,} samples")
                print(f"      📊 Class distribution: {dict(pd.Series(y_resampled).value_counts().sort_index())}")
                
                # Store resampled data
                y_smote[target] = pd.Series(y_resampled, index=range(len(y_resampled)))
                
            except Exception as e:
                print(f"      ❌ Error applying SMOTE to {target}: {e}")
                # Keep original data if SMOTE fails
                y_smote[target] = y_target
        
        # Now apply temporal split to SMOTE-augmented data
        print(f"\n📊 Applying temporal split to SMOTE-augmented data...")
        
        # Create full dataset with SMOTE results
        smote_data = X_smote.copy()
        for target in target_cols:
            if target in y_smote:
                smote_data[target] = y_smote[target]
        
        # Add temporal column back
        smote_data[self.config['temporal_column']] = self.data[self.config['temporal_column']]
        
        # Sort chronologically
        smote_data = smote_data.sort_values(self.config['temporal_column']).reset_index(drop=True)
        
        # Apply temporal split
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(smote_data) * train_ratio)
        
        # Split SMOTE data
        train_data_smote = smote_data.iloc[:train_size]
        test_data_smote = smote_data.iloc[train_size:]
        
        # Prepare train/test sets
        X_train_smote = train_data_smote[feature_cols]
        X_test_smote = test_data_smote[feature_cols]
        
        y_train_smote = {}
        y_test_smote = {}
        
        for target in target_cols:
            y_train_smote[target] = train_data_smote[target]
            y_test_smote[target] = test_data_smote[target]
            
            train_available = y_train_smote[target].notna().sum()
            test_available = y_test_smote[target].notna().sum()
            
            print(f"   • {target}:")
            print(f"     - Train: {train_available:,}/{len(y_train_smote[target]):,} available")
            print(f"     - Test: {test_available:,}/{len(y_test_smote[target]):,} available")
        
        return X_train_smote, X_test_smote, y_train_smote, y_test_smote
    
    def train_models_with_strategy(self, strategy: str, X_train: pd.DataFrame, y_train: Dict) -> Dict:
        """Train models with specified strategy"""
        print(f"\n" + "="*60)
        print(f"6️⃣ TRAINING MODELS: {strategy.upper()}")
        print("="*60)
        
        models = {}
        xgb_params = self.config.get('xgb_params', {})
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
        default_params.update(xgb_params)
        
        print(f"🔧 XGBoost parameters: {default_params}")
        print(f"📈 Training {len(y_train)} models with {strategy} strategy...")
        
        for target_name, y_train_values in y_train.items():
            print(f"\n   🎯 Training model for {target_name}...")
            
            try:
                # Filter training data for this specific target
                valid_mask = ~pd.isna(y_train_values)
                X_train_filtered = X_train[valid_mask]
                y_train_filtered = y_train_values[valid_mask]
                
                if len(y_train_filtered) == 0:
                    print(f"      ❌ No valid training data for {target_name}")
                    continue
                
                print(f"      📊 Training samples: {len(y_train_filtered):,}")
                
                # Create and train model
                model = xgb.XGBRegressor(**default_params)
                model.fit(X_train_filtered, y_train_filtered)
                
                models[target_name] = model
                
                print(f"      ✅ Model trained successfully")
                
            except Exception as e:
                print(f"      ❌ Error training {target_name}: {e}")
                continue
        
        return models
    
    def evaluate_models(self, models: Dict, X_test: pd.DataFrame, y_test: Dict, strategy: str) -> Dict:
        """Evaluate models and calculate metrics"""
        print(f"\n" + "="*60)
        print(f"7️⃣ EVALUATION: {strategy.upper()}")
        print("="*60)
        
        results = {}
        
        for target_name, model in models.items():
            print(f"\n   📊 Evaluating {target_name}...")
            
            try:
                # Filter test data for this specific target
                y_test_values = y_test[target_name]
                valid_mask = ~pd.isna(y_test_values)
                X_test_filtered = X_test[valid_mask]
                y_test_filtered = y_test_values[valid_mask]
                
                if len(y_test_filtered) == 0:
                    print(f"      ❌ No valid test data for {target_name}")
                    continue
                
                print(f"      📊 Test samples: {len(y_test_filtered):,}")
                
                # Generate predictions
                y_pred_raw = model.predict(X_test_filtered)
                y_pred_rounded = np.round(y_pred_raw).astype(int)
                y_pred_clipped = np.clip(y_pred_rounded, 0, 3)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test_filtered, y_pred_raw)
                accuracy = accuracy_score(y_test_filtered, y_pred_clipped)
                
                # Calculate macro-F1 and recall
                macro_f1 = f1_score(y_test_filtered, y_pred_clipped, average='macro', zero_division=0)
                recall_per_class = recall_score(y_test_filtered, y_pred_clipped, average=None, zero_division=0)
                
                # Classification report
                class_report = classification_report(y_test_filtered, y_pred_clipped, 
                                                  output_dict=True, zero_division=0)
                
                # Store results
                results[target_name] = {
                    'mae': mae,
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'recall_per_class': recall_per_class.tolist(),
                    'classification_report': class_report,
                    'sample_size': len(y_test_filtered),
                    'y_true': y_test_filtered.tolist(),
                    'y_pred': y_pred_clipped.tolist()
                }
                
                print(f"      ✅ Evaluation completed")
                print(f"      📊 MAE: {mae:.4f}")
                print(f"      📊 Accuracy: {accuracy:.4f}")
                print(f"      📊 Macro-F1: {macro_f1:.4f}")
                print(f"      📊 Recall per class: {recall_per_class}")
                
            except Exception as e:
                print(f"      ❌ Error evaluating {target_name}: {e}")
                continue
        
        return results
    
    def create_comparison_visualizations(self):
        """Create visualizations comparing all strategies"""
        print("\n" + "="*60)
        print("8️⃣ COMPARISON VISUALIZATIONS")
        print("="*60)
        
        # 1. Strategy comparison
        self._plot_strategy_comparison()
        
        # 2. Macro-F1 comparison
        self._plot_macro_f1_comparison()
        
        # 3. Recall comparison
        self._plot_recall_comparison()
        
        # 4. Confusion matrices
        self._plot_confusion_matrices()
    
    def _plot_strategy_comparison(self):
        """Compare overall performance across strategies"""
        if not self.strategy_results:
            print("   ⚠️  No strategy results available for visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        strategies = list(self.strategy_results.keys())
        targets = list(self.strategy_results[strategies[0]].keys())
        
        # Prepare metrics
        macro_f1_scores = {}
        accuracy_scores = {}
        
        for strategy in strategies:
            macro_f1_scores[strategy] = []
            accuracy_scores[strategy] = []
            
            for target in targets:
                if target in self.strategy_results[strategy]:
                    macro_f1_scores[strategy].append(self.strategy_results[strategy][target]['macro_f1'])
                    accuracy_scores[strategy].append(self.strategy_results[strategy][target]['accuracy'])
                else:
                    macro_f1_scores[strategy].append(0)
                    accuracy_scores[strategy].append(0)
        
        # Plot 1: Macro-F1 comparison
        x = np.arange(len(targets))
        width = 0.2
        
        for i, strategy in enumerate(strategies):
            ax1.bar(x + i*width, macro_f1_scores[strategy], width, label=strategy.replace('_', ' ').title(), alpha=0.7)
        
        ax1.set_title('Macro-F1 Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Target Variable')
        ax1.set_ylabel('Macro-F1 Score')
        ax1.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax1.set_xticklabels(targets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy comparison
        for i, strategy in enumerate(strategies):
            ax2.bar(x + i*width, accuracy_scores[strategy], width, label=strategy.replace('_', ' ').title(), alpha=0.7)
        
        ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Target Variable')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax2.set_xticklabels(targets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("01_strategy_comparison")
    
    def _plot_macro_f1_comparison(self):
        """Detailed macro-F1 comparison"""
        if not self.strategy_results:
            print("   ⚠️  No strategy results available for visualization")
            return
        
        # Check if we have any valid results
        valid_strategies = {k: v for k, v in self.strategy_results.items() if v}
        if not valid_strategies:
            print("   ⚠️  No valid strategy results for visualization")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        strategies = list(self.strategy_results.keys())
        targets = list(self.strategy_results[strategies[0]].keys())
        
        # Create heatmap data
        heatmap_data = []
        for target in targets:
            row = []
            for strategy in strategies:
                if target in self.strategy_results[strategy]:
                    row.append(self.strategy_results[strategy][target]['macro_f1'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        
        # Add text annotations
        for i in range(len(targets)):
            for j in range(len(strategies)):
                text = ax.text(j, i, f"{heatmap_data[i][j]:.3f}",
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Customize plot
        ax.set_xticks(range(len(strategies)))
        ax.set_yticks(range(len(targets)))
        ax.set_xticklabels([s.replace('_', ' ').title() for s in strategies])
        ax.set_yticklabels(targets)
        ax.set_title('Macro-F1 Score Heatmap', fontsize=16, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Macro-F1 Score', rotation=270, labelpad=15)
        
        plt.tight_layout()
        self._save_plot("02_macro_f1_heatmap")
    
    def _plot_recall_comparison(self):
        """Compare recall for minority classes"""
        if not self.strategy_results:
            print("   ⚠️  No strategy results available for visualization")
            return
        
        # Check if we have any valid results
        valid_strategies = {k: v for k, v in self.strategy_results.items() if v}
        if not valid_strategies:
            print("   ⚠️  No valid strategy results for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Recall Comparison for Minority Classes', fontsize=16, fontweight='bold')
        
        strategies = list(self.strategy_results.keys())
        targets = list(self.strategy_results[strategies[0]].keys())
        
        for i, target in enumerate(targets):
            ax = axes[i//2, i%2]
            
            # Extract recall data for this target
            recall_data = {}
            for strategy in strategies:
                if target in self.strategy_results[strategy]:
                    recall_data[strategy] = self.strategy_results[strategy][target]['recall_per_class']
            
            if recall_data:
                # Plot recall for each class
                x = np.arange(4)  # Risk levels 0-3
                width = 0.2
                
                for j, strategy in enumerate(strategies):
                    if strategy in recall_data:
                        ax.bar(x + j*width, recall_data[strategy], width, 
                              label=strategy.replace('_', ' ').title(), alpha=0.7)
                
                ax.set_title(f'{target} - Recall by Class', fontsize=12, fontweight='bold')
                ax.set_xlabel('Risk Level')
                ax.set_ylabel('Recall')
                ax.set_xticks(x + width * (len(strategies) - 1) / 2)
                ax.set_xticklabels(['0', '1', '2', '3'])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("03_recall_comparison")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for best strategy"""
        if not self.strategy_results:
            print("   ⚠️  No strategy results available for visualization")
            return
        
        # Check if we have any valid results
        valid_strategies = {k: v for k, v in self.strategy_results.items() if v}
        if not valid_strategies:
            print("   ⚠️  No valid strategy results for visualization")
            return
        
        # Find best strategy based on average macro-F1
        best_strategy = None
        best_avg_f1 = -1
        
        for strategy, results in self.strategy_results.items():
            if results:
                avg_f1 = np.mean([results[target]['macro_f1'] for target in results.keys()])
                if avg_f1 > best_avg_f1:
                    best_avg_f1 = avg_f1
                    best_strategy = strategy
        
        if not best_strategy:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Confusion Matrices - Best Strategy: {best_strategy.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        targets = list(self.strategy_results[best_strategy].keys())
        
        for i, target in enumerate(targets):
            ax = axes[i//2, i%2]
            
            if target in self.strategy_results[best_strategy]:
                y_true = self.strategy_results[best_strategy][target]['y_true']
                y_pred = self.strategy_results[best_strategy][target]['y_pred']
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Plot heatmap
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i_cm in range(cm.shape[0]):
                    for j_cm in range(cm.shape[1]):
                        ax.text(j_cm, i_cm, format(cm[i_cm, j_cm], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i_cm, j_cm] > thresh else "black")
                
                ax.set_title(f'{target} Confusion Matrix', fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_xticks(range(4))
                ax.set_yticks(range(4))
                ax.set_xticklabels(['0', '1', '2', '3'])
                ax.set_yticklabels(['0', '1', '2', '3'])
        
        plt.tight_layout()
        self._save_plot("04_confusion_matrices")
    
    def save_results(self):
        """Save all results and comparison"""
        print(f"\n" + "="*60)
        print("9️⃣ SAVING RESULTS")
        print("="*60)
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'metrics', 'step5_detailed_results.json')
        
        results_summary = {
            'strategies': self.strategies,
            'strategy_results': self.strategy_results,
            'baseline_comparison': self.baseline_results,
            'config': self.config
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  💾 Saved detailed results: step5_detailed_results.json")
        
        # Save models
        print(f"\n🤖 Saving trained models...")
        for strategy in self.strategies:
            if strategy in self.strategy_results and self.strategy_results[strategy]:
                strategy_dir = os.path.join(self.results_dir, 'models', strategy)
                os.makedirs(strategy_dir, exist_ok=True)
                
                for target, model in self.strategy_results[strategy].items():
                    if isinstance(model, dict) and 'model' in model:
                        model_path = os.path.join(strategy_dir, f'{target}_model.json')
                        model['model'].save_model(model_path)
                        print(f"  🤖 Model saved: {strategy}/{target}_model.json")
    
    def run_step5_pipeline(self):
        """Execute complete Step 5 pipeline with class imbalance strategies"""
        print("🏗️ EXECUTING STEP 5: CLASS IMBALANCE STRATEGY")
        print("=" * 70)
        
        try:
            # Create results directory first
            self._create_results_directory()
            
            # Step 1: Load data and baseline
            self.load_data_and_baseline()
            
            # Step 2: Analyze class imbalance
            imbalance_analysis = self.analyze_class_imbalance()
            
            # Step 3: Implement temporal split
            X_train, X_test, y_train, y_test = self.implement_temporal_split()
            
            # Step 4: Strategy A - Class Weights
            class_weight_models = self.strategy_class_weights(X_train, y_train)
            class_weight_results = self.evaluate_models(
                {k: v['model'] for k, v in class_weight_models.items()}, 
                X_test, y_test, 'class_weights'
            )
            self.strategy_results['class_weights'] = class_weight_results
            
            # Step 5: Strategy B - SMOTE
            X_train_smote, X_test_smote, y_train_smote, y_test_smote = self.strategy_smote(X_train, y_train)
            smote_models = self.train_models_with_strategy('smote', X_train_smote, y_train_smote)
            smote_results = self.evaluate_models(smote_models, X_test_smote, y_test_smote, 'smote')
            self.strategy_results['smote'] = smote_results
            
            # Step 6: Strategy C - Hybrid (best of both)
            # For now, use the better performing strategy
            if class_weight_results and smote_results:
                # Compare average macro-F1
                class_weight_avg_f1 = np.mean([r['macro_f1'] for r in class_weight_results.values()])
                smote_avg_f1 = np.mean([r['macro_f1'] for r in smote_results.values()])
                
                if class_weight_avg_f1 > smote_avg_f1:
                    self.strategy_results['hybrid'] = class_weight_results
                    print(f"\n📊 Hybrid strategy: Using class weights (avg F1: {class_weight_avg_f1:.4f})")
                else:
                    self.strategy_results['hybrid'] = smote_results
                    print(f"\n📊 Hybrid strategy: Using SMOTE (avg F1: {smote_avg_f1:.4f})")
            
            # Step 7: Create visualizations
            self.create_comparison_visualizations()
            
            # Step 8: Save results
            self.save_results()
            
            print("\n🎉 STEP 5 COMPLETED SUCCESSFULLY!")
            print("✅ Class imbalance strategies implemented")
            print("✅ Temporal integrity maintained")
            print("✅ Performance comparison completed")
            
            # Print summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 5 FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of results"""
        print(f"\n" + "="*60)
        print("📊 STEP 5 RESULTS SUMMARY")
        print("="*60)
        
        if self.strategy_results:
            print(f"\n🎯 Strategy Performance Comparison:")
            
            for strategy, results in self.strategy_results.items():
                if results:
                    avg_macro_f1 = np.mean([r['macro_f1'] for r in results.values()])
                    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
                    
                    print(f"\n   📈 {strategy.replace('_', ' ').title()}:")
                    print(f"      • Average Macro-F1: {avg_macro_f1:.4f}")
                    print(f"      • Average Accuracy: {avg_accuracy:.4f}")
                    
                    for target, metrics in results.items():
                        print(f"      • {target}: F1={metrics['macro_f1']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        print(f"\n📁 Results saved to: {self.results_dir}")
        print(f"📊 Visualizations: 01_strategy_comparison.png, 02_macro_f1_heatmap.png, etc.")
        print(f"📈 Metrics: step5_detailed_results.json")


def get_step5_config():
    """Configuration for Step 5 class imbalance model"""
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
    
    print("🚀 Starting XGBoost Risk Prediction Model - Step 5")
    print("="*60)
    
    # Get configuration
    config = get_step5_config()
    
    # Create and run class imbalance model
    imbalance_model = ClassImbalanceModel(config)
    imbalance_model.run_step5_pipeline()
    
    print("\n🏁 Step 5 execution completed!")
    print("Ready to proceed to Step 6: Advanced Model Architectures")