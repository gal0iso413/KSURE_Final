"""
Step 5: Class Imbalance Strategy
================================
Compare different imbalance strategies to choose the best approach for handling
severe class imbalance in credit risk prediction.

Strategies:
1. Baseline (No imbalance strategy) - from Step 4
2. Class Weights (sample_weight for regression)
3. SMOTE (after temporal split, no data leakage)

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Visualization
plt.style.use('default')
sns.set_palette("husl")

class ClassImbalanceModel:
    """Class Imbalance Strategy Comparison for Credit Risk Prediction"""
    
    def __init__(self):
        """Initialize the model with configuration"""
        self.config = {
            'data_path': 'dataset/credit_risk_dataset.csv',
            'temporal_column': '보험청약일자',
            'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
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
            'train_ratio': 0.8,
            'random_state': 42,
            'xgb_params': {
                'enable_missing': True,
                'random_state': 42,
                'verbosity': 0
            }
        }
        
        # Initialize data containers
        self.data = None
        self.baseline_results = {}
        self.strategy_results = {}
        self.best_strategy = None
        
        # Create results directory
        self.results_dir = 'result/step5_imbalance'
        self._create_results_directory()
        
    def _create_results_directory(self):
        """Create results directory if it doesn't exist"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.results_dir}/models', exist_ok=True)
        
    def load_data_and_baseline(self) -> Tuple[pd.DataFrame, Dict]:
        """Load data and Step 4 baseline results"""
        print("\n" + "="*60)
        print("1️⃣ DATA LOADING & BASELINE ANALYSIS")
        print("="*60)
        
        # Load dataset
        print("📊 Loading dataset...")
        self.data = pd.read_csv(self.config['data_path'])
        print(f"✅ Dataset loaded: {len(self.data):,} records, {len(self.data.columns)} features")
        
        # Load Step 4 baseline results
        baseline_path = 'result/step4_temporal/step4_results.json'
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r', encoding='utf-8') as f:
                self.baseline_results = json.load(f)
            print("✅ Step 4 baseline results loaded")
        else:
            print("⚠️  Step 4 results not found. Will create baseline from scratch.")
            self.baseline_results = {}
        
        # Temporal column setup
        temporal_col = self.config['temporal_column']
        if temporal_col in self.data.columns:
            self.data[temporal_col] = pd.to_datetime(self.data[temporal_col], errors='coerce')
            print(f"✅ Temporal column converted: {temporal_col}")
        else:
            raise ValueError(f"Temporal column '{temporal_col}' not found in dataset")
        
        return self.data, self.baseline_results
    
    def implement_temporal_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Implement temporal train/test split (exact same as Step 4)"""
        print("\n" + "="*60)
        print("2️⃣ TEMPORAL SPLIT (EXACT SAME AS STEP 4)")
        print("="*60)
        
        # Sort data chronologically by temporal column
        temporal_col = self.config['temporal_column']
        self.data = self.data.sort_values(temporal_col).reset_index(drop=True)
        
        # Convert temporal column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data[temporal_col]):
            self.data[temporal_col] = pd.to_datetime(self.data[temporal_col])
        
        # Define cutoff dates for complete prediction periods (same as Step 4)
        current_date = pd.Timestamp('2025-06-30')
        
        # Calculate cutoff dates for each risk year
        cutoff_dates = {
            'risk_year1': current_date - pd.DateOffset(years=1),  # 2024-06-30
            'risk_year2': current_date - pd.DateOffset(years=2),  # 2023-06-30  
            'risk_year3': current_date - pd.DateOffset(years=3),  # 2022-06-30
            'risk_year4': current_date - pd.DateOffset(years=4)   # 2021-06-30
        }
        
        print(f"📊 Temporal split configuration:")
        print(f"   • Current date: {current_date.strftime('%Y-%m-%d')}")
        print(f"   • Cutoff dates for complete prediction periods:")
        for target, cutoff in cutoff_dates.items():
            print(f"     - {target}: {cutoff.strftime('%Y-%m-%d')}")
        
        # Use the most restrictive cutoff (risk_year4) for the main split
        main_cutoff = cutoff_dates['risk_year4']  # 2021-06-30
        
        # Filter data to only include contracts before the cutoff (same as Step 4)
        valid_data = self.data[self.data[temporal_col] <= main_cutoff].copy()
        
        print(f"\n📊 Data filtering results:")
        print(f"   • Original data: {len(self.data):,} records")
        print(f"   • Valid data (before {main_cutoff.strftime('%Y-%m-%d')}): {len(valid_data):,} records")
        print(f"   • Excluded data: {len(self.data) - len(valid_data):,} records")
        
        # Simple chronological 80/20 split on valid data (same as Step 4)
        train_ratio = self.config.get('train_ratio', 0.8)
        train_size = int(len(valid_data) * train_ratio)
        
        print(f"\n📊 Temporal split details:")
        print(f"   • Train records: {train_size:,} ({train_size/len(valid_data):.1%})")
        print(f"   • Test records: {len(valid_data) - train_size:,} ({(len(valid_data) - train_size)/len(valid_data):.1%})")
        
        # Get the actual split date for reporting
        split_date = valid_data.iloc[train_size-1][temporal_col]
        print(f"   • Split date: {split_date.strftime('%Y-%m-%d')}")
        
        # Split data
        train_data = valid_data.iloc[:train_size].copy()
        test_data = valid_data.iloc[train_size:].copy()
        
        # Prepare features and targets
        exclude_cols = self.config.get('exclude_columns', [])
        target_cols = self.config['target_columns']
        
        # Feature columns (exclude targets, excluded columns, and temporal column)
        feature_cols = [col for col in valid_data.columns 
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
        
        return X_train, X_test, y_train, y_test
    
    def strategy_baseline(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 1: No imbalance strategy (baseline from Step 4)"""
        print("\n" + "="*60)
        print("3️⃣ STRATEGY 1: BASELINE (NO IMBALANCE STRATEGY)")
        print("="*60)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Training baseline model for {target}...")
            
            # Get valid data for this target
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask].copy()
            y_train_valid = y_train[target][train_mask].copy()
            X_test_valid = X_test[test_mask].copy()
            y_test_valid = y_test[target][test_mask].copy()
            
            print(f"   📊 Training samples: {len(X_train_valid):,}")
            print(f"   📊 Test samples: {len(X_test_valid):,}")
            
            # Train model (no imbalance strategy)
            model = xgb.XGBRegressor(**self.config['xgb_params'])
            model.fit(X_train_valid, y_train_valid)
            
            # Predictions
            y_pred = model.predict(X_test_valid)
            y_pred_rounded = np.round(y_pred).astype(int)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_valid, y_pred)
            accuracy = accuracy_score(y_test_valid, y_pred_rounded)
            
            # Classification metrics
            f1_macro = f1_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            recall_avg = recall_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            
            # Per-class recall
            recall_per_class = recall_score(y_test_valid, y_pred_rounded, average=None, zero_division=0)
            
            results[target] = {
                'mae': mae,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'recall_avg': recall_avg,
                'recall_per_class': recall_per_class.tolist(),
                'predictions': y_pred_rounded.tolist(),
                'actual': y_test_valid.tolist()
            }
            
            print(f"   ✅ {target} - MAE: {mae:.4f}, F1-Macro: {f1_macro:.4f}, Recall: {recall_avg:.4f}")
        
        return results
    
    def strategy_class_weights(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 2: Class weights using sample_weight for regression"""
        print("\n" + "="*60)
        print("4️⃣ STRATEGY 2: CLASS WEIGHTS (SAMPLE_WEIGHT)")
        print("="*60)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Training class weights model for {target}...")
            
            # Get valid data for this target
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask].copy()
            y_train_valid = y_train[target][train_mask].copy()
            X_test_valid = X_test[test_mask].copy()
            y_test_valid = y_test[target][test_mask].copy()
            
            print(f"   📊 Training samples: {len(X_train_valid):,}")
            print(f"   📊 Test samples: {len(X_test_valid):,}")
            
            # Calculate class weights using sample_weight
            unique_classes = np.array(sorted(y_train_valid.unique()))
            class_weights = compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=y_train_valid
            )
            
            # Create sample weights
            class_weight_dict = dict(zip(unique_classes, class_weights))
            sample_weights = [class_weight_dict[int(y)] for y in y_train_valid]
            
            print(f"   📊 Class weights: {dict(zip(unique_classes, class_weights))}")
            
            # Train model with sample weights
            model = xgb.XGBRegressor(**self.config['xgb_params'])
            model.fit(X_train_valid, y_train_valid, sample_weight=sample_weights)
            
            # Predictions
            y_pred = model.predict(X_test_valid)
            y_pred_rounded = np.round(y_pred).astype(int)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_valid, y_pred)
            accuracy = accuracy_score(y_test_valid, y_pred_rounded)
            f1_macro = f1_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            recall_avg = recall_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            recall_per_class = recall_score(y_test_valid, y_pred_rounded, average=None, zero_division=0)
            
            results[target] = {
                'mae': mae,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'recall_avg': recall_avg,
                'recall_per_class': recall_per_class.tolist(),
                'predictions': y_pred_rounded.tolist(),
                'actual': y_test_valid.tolist()
            }
            
            print(f"   ✅ {target} - MAE: {mae:.4f}, F1-Macro: {f1_macro:.4f}, Recall: {recall_avg:.4f}")
        
        return results
    
    def strategy_smote(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 3: SMOTE applied after temporal split (no data leakage)"""
        print("\n" + "="*60)
        print("5️⃣ STRATEGY 3: SMOTE (AFTER TEMPORAL SPLIT)")
        print("="*60)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Training SMOTE model for {target}...")
            
            # Get valid data for this target
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask].copy()
            y_train_valid = y_train[target][train_mask].copy()
            X_test_valid = X_test[test_mask].copy()
            y_test_valid = y_test[target][test_mask].copy()
            
            print(f"   📊 Original training samples: {len(X_train_valid):,}")
            print(f"   📊 Test samples: {len(X_test_valid):,}")
            
            # Apply SMOTE to training data only (no data leakage)
            try:
                # Handle NaN values in features
                imputer = SimpleImputer(strategy='median')
                X_train_imputed = imputer.fit_transform(X_train_valid)
                
                # Apply SMOTE
                smote = SMOTE(random_state=self.config['random_state'], k_neighbors=5)
                X_train_smote, y_train_smote = smote.fit_resample(X_train_imputed, y_train_valid)
                
                print(f"   📊 After SMOTE: {len(X_train_smote):,} training samples")
                print(f"   📊 SMOTE augmentation: +{len(X_train_smote) - len(X_train_valid):,} samples")
                
                # Train model with SMOTE-augmented data
                model = xgb.XGBRegressor(**self.config['xgb_params'])
                model.fit(X_train_smote, y_train_smote)
                
            except Exception as e:
                print(f"   ⚠️  SMOTE failed for {target}: {str(e)}")
                print(f"   🔄 Falling back to original training data")
                
                # Fallback to original data
                model = xgb.XGBRegressor(**self.config['xgb_params'])
                model.fit(X_train_valid, y_train_valid)
            
            # Predictions
            y_pred = model.predict(X_test_valid)
            y_pred_rounded = np.round(y_pred).astype(int)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_valid, y_pred)
            accuracy = accuracy_score(y_test_valid, y_pred_rounded)
            f1_macro = f1_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            recall_avg = recall_score(y_test_valid, y_pred_rounded, average='macro', zero_division=0)
            recall_per_class = recall_score(y_test_valid, y_pred_rounded, average=None, zero_division=0)
            
            results[target] = {
                'mae': mae,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'recall_avg': recall_avg,
                'recall_per_class': recall_per_class.tolist(),
                'predictions': y_pred_rounded.tolist(),
                'actual': y_test_valid.tolist()
            }
            
            print(f"   ✅ {target} - MAE: {mae:.4f}, F1-Macro: {f1_macro:.4f}, Recall: {recall_avg:.4f}")
        
        return results
    
    def compare_strategies(self) -> str:
        """Compare all strategies and select the best one"""
        print("\n" + "="*60)
        print("6️⃣ STRATEGY COMPARISON & SELECTION")
        print("="*60)
        
        # Calculate average performance for each strategy
        strategy_scores = {}
        
        for strategy_name, results in self.strategy_results.items():
            if not results:
                continue
                
            f1_scores = [results[target]['f1_macro'] for target in self.config['target_columns']]
            recall_scores = [results[target]['recall_avg'] for target in self.config['target_columns']]
            
            avg_f1 = np.mean(f1_scores)
            avg_recall = np.mean(recall_scores)
            
            strategy_scores[strategy_name] = {
                'avg_f1_macro': avg_f1,
                'avg_recall': avg_recall,
                'f1_scores': f1_scores,
                'recall_scores': recall_scores
            }
            
            print(f"\n📊 {strategy_name.upper()} Performance:")
            print(f"   • Average F1-Macro: {avg_f1:.4f}")
            print(f"   • Average Recall: {avg_recall:.4f}")
            
            for i, target in enumerate(self.config['target_columns']):
                print(f"   • {target}: F1={f1_scores[i]:.4f}, Recall={recall_scores[i]:.4f}")
        
        # Select best strategy based on F1-Macro (primary metric)
        if strategy_scores:
            best_strategy = max(strategy_scores.keys(), 
                              key=lambda x: strategy_scores[x]['avg_f1_macro'])
            
            print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
            print(f"   • Average F1-Macro: {strategy_scores[best_strategy]['avg_f1_macro']:.4f}")
            print(f"   • Average Recall: {strategy_scores[best_strategy]['avg_recall']:.4f}")
            
            # Improvement over baseline
            if 'baseline' in strategy_scores:
                baseline_f1 = strategy_scores['baseline']['avg_f1_macro']
                improvement = ((strategy_scores[best_strategy]['avg_f1_macro'] - baseline_f1) / baseline_f1) * 100
                print(f"   • Improvement over baseline: {improvement:+.2f}%")
            
            self.best_strategy = best_strategy
            return best_strategy
        else:
            print("❌ No valid strategy results to compare")
            return None
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        print("\n" + "="*60)
        print("7️⃣ CREATING VISUALIZATIONS")
        print("="*60)
        
        if not self.strategy_results:
            print("⚠️  No strategy results available for visualization")
            return
        
        # 1. F1-Macro Comparison
        self._plot_f1_comparison()
        
        # 2. Recall Comparison
        self._plot_recall_comparison()
        
        # 3. Confusion Matrix for Best Strategy
        self._plot_confusion_matrices()
        
        print("✅ All visualizations created successfully")
    
    def _plot_f1_comparison(self):
        """Plot F1-Macro comparison across strategies"""
        strategies = list(self.strategy_results.keys())
        targets = self.config['target_columns']
        
        # Prepare data
        f1_data = []
        for strategy in strategies:
            if strategy in self.strategy_results:
                for target in targets:
                    f1_data.append({
                        'Strategy': strategy.replace('_', ' ').title(),
                        'Target': target,
                        'F1-Macro': self.strategy_results[strategy][target]['f1_macro']
                    })
        
        if not f1_data:
            return
        
        df_f1 = pd.DataFrame(f1_data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_f1, x='Target', y='F1-Macro', hue='Strategy')
        plt.title('F1-Macro Score Comparison Across Strategies', fontsize=14, fontweight='bold')
        plt.xlabel('Target Variable', fontsize=12)
        plt.ylabel('F1-Macro Score', fontsize=12)
        plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/f1_macro_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ F1-Macro comparison plot saved")
    
    def _plot_recall_comparison(self):
        """Plot recall comparison for minority classes"""
        strategies = list(self.strategy_results.keys())
        targets = self.config['target_columns']
        
        # Prepare data for minority classes (1, 2, 3)
        recall_data = []
        for strategy in strategies:
            if strategy in self.strategy_results:
                for target in targets:
                    recall_per_class = self.strategy_results[strategy][target]['recall_per_class']
                    for class_idx, recall in enumerate(recall_per_class):
                        if class_idx > 0:  # Only minority classes
                            recall_data.append({
                                'Strategy': strategy.replace('_', ' ').title(),
                                'Target': target,
                                'Class': f'Class {class_idx}',
                                'Recall': recall
                            })
        
        if not recall_data:
            return
        
        df_recall = pd.DataFrame(recall_data)
        
        # Create plot using catplot (supports col parameter)
        plt.figure(figsize=(16, 10))
        g = sns.catplot(
            data=df_recall, 
            x='Target', 
            y='Recall', 
            hue='Strategy', 
            col='Class',
            kind='bar',
            height=4,
            aspect=1.2
        )
        g.fig.suptitle('Recall for Minority Classes Across Strategies', fontsize=16, fontweight='bold', y=1.02)
        g.fig.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/recall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Recall comparison plot saved")
    
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for best strategy"""
        if not self.best_strategy or self.best_strategy not in self.strategy_results:
            return
        
        results = self.strategy_results[self.best_strategy]
        targets = self.config['target_columns']
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Confusion Matrices - Best Strategy: {self.best_strategy.upper()}', 
                    fontsize=16, fontweight='bold')
        
        for i, target in enumerate(targets):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Get predictions and actual values
            y_pred = results[target]['predictions']
            y_actual = results[target]['actual']
            
            # Create confusion matrix
            cm = confusion_matrix(y_actual, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{target}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Confusion matrices plot saved")
    
    def save_results(self):
        """Save all results and summary"""
        print("\n" + "="*60)
        print("8️⃣ SAVING RESULTS")
        print("="*60)
        
        # Prepare results summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'best_strategy': self.best_strategy,
            'strategy_comparison': {},
            'target_performance': {},
            'configuration': self.config
        }
        
        # Strategy comparison
        for strategy_name, results in self.strategy_results.items():
            if results:
                f1_scores = [results[target]['f1_macro'] for target in self.config['target_columns']]
                recall_scores = [results[target]['recall_avg'] for target in self.config['target_columns']]
                
                summary['strategy_comparison'][strategy_name] = {
                    'avg_f1_macro': np.mean(f1_scores),
                    'avg_recall': np.mean(recall_scores),
                    'f1_scores': f1_scores,
                    'recall_scores': recall_scores
                }
        
        # Target performance for best strategy
        if self.best_strategy and self.best_strategy in self.strategy_results:
            best_results = self.strategy_results[self.best_strategy]
            for target in self.config['target_columns']:
                summary['target_performance'][target] = {
                    'mae': best_results[target]['mae'],
                    'accuracy': best_results[target]['accuracy'],
                    'f1_macro': best_results[target]['f1_macro'],
                    'recall_avg': best_results[target]['recall_avg'],
                    'recall_per_class': best_results[target]['recall_per_class']
                }
        
        # Save summary
        with open(f'{self.results_dir}/step5_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        with open(f'{self.results_dir}/step5_detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.strategy_results, f, indent=2, ensure_ascii=False)
        
        print("✅ Results saved successfully")
        print(f"   📁 Summary: {self.results_dir}/step5_results.json")
        print(f"   📁 Detailed: {self.results_dir}/step5_detailed_results.json")
        
        # Print final summary
        print(f"\n🎯 STEP 5 COMPLETED SUCCESSFULLY")
        print(f"   🏆 Best Strategy: {self.best_strategy.upper()}")
        if self.best_strategy and 'baseline' in summary['strategy_comparison']:
            baseline_f1 = summary['strategy_comparison']['baseline']['avg_f1_macro']
            best_f1 = summary['strategy_comparison'][self.best_strategy]['avg_f1_macro']
            improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
            print(f"   📈 Improvement over baseline: {improvement:+.2f}%")
    
    def run_step5_pipeline(self):
        """Run the complete Step 5 pipeline"""
        print("🚀 STEP 5: CLASS IMBALANCE STRATEGY COMPARISON")
        print("="*60)
        print("Goal: Compare imbalance strategies and select the best approach")
        print("="*60)
        
        try:
            # Step 1: Load data and baseline
            self.data, self.baseline_results = self.load_data_and_baseline()
            
            # Step 2: Implement temporal split
            X_train, X_test, y_train, y_test = self.implement_temporal_split()
            
            # Step 3: Strategy 1 - Baseline (no imbalance strategy)
            self.strategy_results['baseline'] = self.strategy_baseline(X_train, X_test, y_train, y_test)
            
            # Step 4: Strategy 2 - Class weights
            self.strategy_results['class_weights'] = self.strategy_class_weights(X_train, X_test, y_train, y_test)
            
            # Step 5: Strategy 3 - SMOTE
            self.strategy_results['smote'] = self.strategy_smote(X_train, X_test, y_train, y_test)
            
            # Step 6: Compare strategies and select best
            self.compare_strategies()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            # Step 8: Save results
            self.save_results()
            
            print(f"\n🎉 STEP 5 COMPLETED SUCCESSFULLY!")
            print(f"   📊 Compared 3 imbalance strategies")
            print(f"   🏆 Selected best strategy: {self.best_strategy.upper()}")
            print(f"   📁 Results saved to: {self.results_dir}")
            
        except Exception as e:
            print(f"\n❌ ERROR in Step 5: {str(e)}")
            raise

def main():
    """Main execution function"""
    model = ClassImbalanceModel()
    model.run_step5_pipeline()

if __name__ == "__main__":
    main()