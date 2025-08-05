"""
Step 5-2: Advanced Class Imbalance Techniques
==============================================
Implement advanced techniques to dramatically improve high-risk detection:
1. Advanced Sampling Strategies (ADASYN, BorderlineSMOTE, etc.)
2. Ensemble Methods with diverse sampling approaches
3. Binary Cascade Architecture (Risk Detection → Risk Level Classification)
4. Custom ensemble weighting for high-risk detection

Goal: Achieve 15-25% high-risk recall (vs current 5.6%)

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
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.metrics import (f1_score, recall_score, precision_score, 
                           accuracy_score, confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# Advanced Sampling
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# Visualization
plt.style.use('default')
sns.set_palette("husl")

class AdvancedImbalanceModel:
    """Advanced Class Imbalance Techniques for Extreme Imbalance"""
    
    def __init__(self):
        """Initialize with configuration"""
        self.config = {
            'data_path': 'dataset/credit_risk_dataset.csv',
            'temporal_column': '보험청약일자',
            'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
            'exclude_columns': [
                '사업자등록번호', '대상자명', '대상자등록이력일시', '대상자기본주소',
                '청약번호', '청약상태코드', '수출자대상자번호', '특별출연협약코드', '업종코드1'
            ],
            'train_ratio': 0.8,
            'random_state': 42,
            'xgb_base_params': {
                'random_state': 42,
                'verbosity': 0,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            }
        }
        
        # Initialize containers
        self.data = None
        self.results = {}
        self.best_approach = None
        
        # Create results directory
        self.results_dir = 'result/step5_advanced'
        self._create_results_directory()
        
    def _create_results_directory(self):
        """Create directory structure"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/plots', exist_ok=True)
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Load data with same preprocessing as Step 5-1"""
        print("🚀 ADVANCED CLASS IMBALANCE TECHNIQUES")
        print("="*60)
        print("Goal: Achieve 15-25% high-risk recall (4x improvement)")
        print("="*60)
        
        print("\n1️⃣ DATA LOADING & PREPARATION")
        print("-" * 40)
        
        # Load and process data (same as 5-1)
        self.data = pd.read_csv(self.config['data_path'])
        print(f"✅ Dataset loaded: {len(self.data):,} records")
        
        # Convert temporal column
        self.data[self.config['temporal_column']] = pd.to_datetime(
            self.data[self.config['temporal_column']], format='%Y-%m-%d'
        )
        
        # Temporal split logic (same as previous)
        cutoff_dates = {}
        for i, target in enumerate(self.config['target_columns'], 1):
            cutoff_date = datetime(2025 - i, 6, 30).date()
            cutoff_dates[target] = cutoff_date
        
        earliest_cutoff = min(cutoff_dates.values())
        valid_data = self.data[
            self.data[self.config['temporal_column']].dt.date < earliest_cutoff
        ].copy()
        
        # Split data
        split_index = int(len(valid_data) * self.config['train_ratio'])
        valid_data_sorted = valid_data.sort_values(self.config['temporal_column'])
        
        train_data = valid_data_sorted.iloc[:split_index]
        test_data = valid_data_sorted.iloc[split_index:]
        
        print(f"📊 Train: {len(train_data):,} | Test: {len(test_data):,}")
        
        # Prepare features
        feature_columns = [
            col for col in self.data.columns 
            if col not in self.config['target_columns'] + self.config['exclude_columns'] + [self.config['temporal_column']]
        ]
        
        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]
        y_train = {target: train_data[target] for target in self.config['target_columns']}
        y_test = {target: test_data[target] for target in self.config['target_columns']}
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        print(f"📊 Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def advanced_sampling_strategy(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 1: Advanced Sampling Techniques"""
        print(f"\n2️⃣ STRATEGY 1: ADVANCED SAMPLING TECHNIQUES")
        print("-" * 40)
        
        # Define sampling strategies
        sampling_strategies = {
            'ADASYN': ADASYN(random_state=self.config['random_state'], n_neighbors=3),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=self.config['random_state'], k_neighbors=3),
            'SVMSMOTE': SVMSMOTE(random_state=self.config['random_state'], k_neighbors=3),
            'SMOTEENN': SMOTEENN(random_state=self.config['random_state']),
            'SMOTETomek': SMOTETomek(random_state=self.config['random_state'])
        }
        
        results = {}
        
        for strategy_name, sampler in sampling_strategies.items():
            print(f"\n🔬 Testing {strategy_name}...")
            strategy_results = {}
            
            for target in self.config['target_columns']:
                # Get valid data
                train_mask = y_train[target].notna()
                test_mask = y_test[target].notna()
                
                X_train_valid = X_train[train_mask]
                y_train_valid = y_train[target][train_mask].astype(int)
                X_test_valid = X_test[test_mask]
                y_test_valid = y_test[target][test_mask].astype(int)
                
                try:
                    # Apply sampling
                    X_resampled, y_resampled = sampler.fit_resample(X_train_valid, y_train_valid)
                    
                    print(f"   📈 {target}: {len(X_train_valid)} → {len(X_resampled)} samples")
                    
                    # Train model
                    model = xgb.XGBClassifier(**self.config['xgb_base_params'])
                    model.fit(X_resampled, y_resampled)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_test_valid)
                    
                    # Calculate metrics
                    f1_macro = f1_score(y_test_valid, y_pred, average='macro', zero_division=0)
                    recall_macro = recall_score(y_test_valid, y_pred, average='macro', zero_division=0)
                    
                    # High-risk metrics
                    high_risk_mask = (y_test_valid >= 2)
                    high_risk_pred = (y_pred >= 2)
                    
                    if np.any(high_risk_mask):
                        high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                        high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
                    else:
                        high_risk_recall = high_risk_f1 = 0.0
                    
                    strategy_results[target] = {
                        'f1_macro': f1_macro,
                        'recall_macro': recall_macro,
                        'high_risk_f1': high_risk_f1,
                        'high_risk_recall': high_risk_recall,
                        'samples_generated': len(X_resampled) - len(X_train_valid)
                    }
                    
                except Exception as e:
                    print(f"   ❌ {target} failed: {str(e)}")
                    strategy_results[target] = {
                        'f1_macro': 0.0, 'recall_macro': 0.0, 
                        'high_risk_f1': 0.0, 'high_risk_recall': 0.0,
                        'samples_generated': 0
                    }
            
            results[strategy_name] = strategy_results
            
            # Print summary
            avg_hr_recall = np.mean([strategy_results[t]['high_risk_recall'] for t in self.config['target_columns']])
            print(f"   🎯 Average High-Risk Recall: {avg_hr_recall:.1%}")
        
        return results
    
    def ensemble_strategy(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 2: Ensemble with Diverse Sampling"""
        print(f"\n3️⃣ STRATEGY 2: ENSEMBLE WITH DIVERSE SAMPLING")
        print("-" * 40)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Building ensemble for {target}...")
            
            # Get valid data
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask]
            y_train_valid = y_train[target][train_mask].astype(int)
            X_test_valid = X_test[test_mask]
            y_test_valid = y_test[target][test_mask].astype(int)
            
            # Create ensemble models with different sampling strategies
            ensemble_models = []
            
            try:
                # Model 1: ADASYN sampling
                adasyn = ADASYN(random_state=42, n_neighbors=3)
                X_ada, y_ada = adasyn.fit_resample(X_train_valid, y_train_valid)
                model1 = xgb.XGBClassifier(**self.config['xgb_base_params'], random_state=42)
                model1.fit(X_ada, y_ada)
                ensemble_models.append(('adasyn', model1))
                
                # Model 2: BorderlineSMOTE
                borderline = BorderlineSMOTE(random_state=43, k_neighbors=3)
                X_border, y_border = borderline.fit_resample(X_train_valid, y_train_valid)
                model2 = xgb.XGBClassifier(**self.config['xgb_base_params'], random_state=43)
                model2.fit(X_border, y_border)
                ensemble_models.append(('borderline', model2))
                
                # Model 3: Focal Loss (from Step 5-1)
                classes = np.unique(y_train_valid)
                class_counts = np.bincount(y_train_valid)
                focal_weights = []
                for class_idx in classes:
                    class_freq = class_counts[class_idx] / len(y_train_valid)
                    weight = (1.0 / class_freq) ** 0.5
                    focal_weights.append(weight)
                focal_weights = np.array(focal_weights) / np.mean(focal_weights)
                sample_weights = np.array([focal_weights[int(y)] for y in y_train_valid])
                
                model3 = xgb.XGBClassifier(**self.config['xgb_base_params'], random_state=44)
                model3.fit(X_train_valid, y_train_valid, sample_weight=sample_weights)
                ensemble_models.append(('focal', model3))
                
                print(f"   ✅ Built ensemble with {len(ensemble_models)} models")
                
                # Ensemble prediction with custom weighting
                ensemble_predictions = []
                weights = [0.4, 0.4, 0.2]  # Higher weight for sampling methods
                
                for model_name, model in ensemble_models:
                    pred_proba = model.predict_proba(X_test_valid)
                    ensemble_predictions.append(pred_proba)
                
                # Weighted average of probabilities
                weighted_proba = np.zeros_like(ensemble_predictions[0])
                for i, (weight, pred_proba) in enumerate(zip(weights, ensemble_predictions)):
                    weighted_proba += weight * pred_proba
                
                # Final prediction
                y_pred_ensemble = np.argmax(weighted_proba, axis=1)
                
            except Exception as e:
                print(f"   ❌ Ensemble failed: {str(e)}")
                y_pred_ensemble = np.zeros(len(y_test_valid))
            
            # Calculate metrics
            f1_macro = f1_score(y_test_valid, y_pred_ensemble, average='macro', zero_division=0)
            recall_macro = recall_score(y_test_valid, y_pred_ensemble, average='macro', zero_division=0)
            
            # High-risk metrics
            high_risk_mask = (y_test_valid >= 2)
            high_risk_pred = (y_pred_ensemble >= 2)
            
            if np.any(high_risk_mask):
                high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
            else:
                high_risk_recall = high_risk_f1 = 0.0
            
            results[target] = {
                'f1_macro': f1_macro,
                'recall_macro': recall_macro,
                'high_risk_f1': high_risk_f1,
                'high_risk_recall': high_risk_recall,
                'ensemble_size': len(ensemble_models)
            }
            
            print(f"   📊 F1: {f1_macro:.3f}, HR-Recall: {high_risk_recall:.1%}")
        
        return results
    
    def binary_cascade_strategy(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                               y_train: Dict, y_test: Dict) -> Dict:
        """Strategy 3: Binary Cascade (Risk Detection → Risk Classification)"""
        print(f"\n4️⃣ STRATEGY 3: BINARY CASCADE ARCHITECTURE")
        print("-" * 40)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Building cascade for {target}...")
            
            # Get valid data
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask]
            y_train_valid = y_train[target][train_mask].astype(int)
            X_test_valid = X_test[test_mask]
            y_test_valid = y_test[target][test_mask].astype(int)
            
            try:
                # Stage 1: Binary Risk Detection (0 vs 1,2,3)
                y_binary_train = (y_train_valid > 0).astype(int)
                y_binary_test = (y_test_valid > 0).astype(int)
                
                # Apply ADASYN for binary classification
                adasyn_binary = ADASYN(random_state=42, n_neighbors=5)
                X_binary_resampled, y_binary_resampled = adasyn_binary.fit_resample(
                    X_train_valid, y_binary_train
                )
                
                # Train binary classifier
                binary_model = xgb.XGBClassifier(
                    **self.config['xgb_base_params'], 
                    random_state=42
                )
                binary_model.fit(X_binary_resampled, y_binary_resampled)
                
                # Binary predictions
                binary_pred = binary_model.predict(X_test_valid)
                binary_proba = binary_model.predict_proba(X_test_valid)[:, 1]
                
                print(f"   📊 Stage 1 - Risk Detection Recall: {recall_score(y_binary_test, binary_pred):.1%}")
                
                # Stage 2: Risk Level Classification (only for predicted risk cases)
                risk_indices = np.where(binary_pred == 1)[0]
                
                if len(risk_indices) > 0:
                    # Get risk cases from training data
                    risk_mask_train = y_train_valid > 0
                    X_risk_train = X_train_valid[risk_mask_train]
                    y_risk_train = y_train_valid[risk_mask_train]
                    
                    if len(np.unique(y_risk_train)) > 1:
                        # Apply sampling for risk level classification
                        try:
                            borderline_risk = BorderlineSMOTE(random_state=42, k_neighbors=3)
                            X_risk_resampled, y_risk_resampled = borderline_risk.fit_resample(
                                X_risk_train, y_risk_train
                            )
                        except:
                            X_risk_resampled, y_risk_resampled = X_risk_train, y_risk_train
                        
                        # Train risk level classifier
                        risk_model = xgb.XGBClassifier(
                            **self.config['xgb_base_params'], 
                            random_state=43
                        )
                        risk_model.fit(X_risk_resampled, y_risk_resampled)
                        
                        # Predict risk levels for detected risk cases
                        X_risk_test = X_test_valid.iloc[risk_indices]
                        risk_level_pred = risk_model.predict(X_risk_test)
                    else:
                        # If only one risk level in training, predict that level
                        risk_level_pred = np.full(len(risk_indices), y_risk_train.iloc[0])
                else:
                    risk_level_pred = np.array([])
                
                # Combine predictions
                y_pred_cascade = np.zeros(len(y_test_valid))
                if len(risk_indices) > 0:
                    y_pred_cascade[risk_indices] = risk_level_pred
                
                print(f"   📊 Stage 2 - Classified {len(risk_indices)} risk cases")
                
            except Exception as e:
                print(f"   ❌ Cascade failed: {str(e)}")
                y_pred_cascade = np.zeros(len(y_test_valid))
            
            # Calculate metrics
            f1_macro = f1_score(y_test_valid, y_pred_cascade, average='macro', zero_division=0)
            recall_macro = recall_score(y_test_valid, y_pred_cascade, average='macro', zero_division=0)
            
            # High-risk metrics
            high_risk_mask = (y_test_valid >= 2)
            high_risk_pred = (y_pred_cascade >= 2)
            
            if np.any(high_risk_mask):
                high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
            else:
                high_risk_recall = high_risk_f1 = 0.0
            
            results[target] = {
                'f1_macro': f1_macro,
                'recall_macro': recall_macro,
                'high_risk_f1': high_risk_f1,
                'high_risk_recall': high_risk_recall,
                'binary_accuracy': accuracy_score(y_binary_test, binary_pred) if 'binary_pred' in locals() else 0.0
            }
            
            print(f"   📊 Final - F1: {f1_macro:.3f}, HR-Recall: {high_risk_recall:.1%}")
        
        return results
    
    def compare_all_strategies(self, sampling_results: Dict, ensemble_results: Dict, 
                              cascade_results: Dict):
        """Compare all advanced strategies"""
        print(f"\n5️⃣ STRATEGY COMPARISON & SELECTION")
        print("=" * 60)
        
        all_strategies = {
            'Advanced_Sampling': sampling_results,
            'Ensemble': ensemble_results,
            'Binary_Cascade': cascade_results
        }
        
        comparison_data = []
        
        # For sampling results, find best sampling method
        if sampling_results:
            best_sampling = None
            best_sampling_score = 0
            
            for method_name, method_results in sampling_results.items():
                avg_hr_recall = np.mean([method_results[t]['high_risk_recall'] for t in self.config['target_columns']])
                if avg_hr_recall > best_sampling_score:
                    best_sampling_score = avg_hr_recall
                    best_sampling = method_name
            
            if best_sampling:
                best_results = sampling_results[best_sampling]
                avg_f1 = np.mean([best_results[t]['f1_macro'] for t in self.config['target_columns']])
                avg_recall = np.mean([best_results[t]['recall_macro'] for t in self.config['target_columns']])
                avg_hr_f1 = np.mean([best_results[t]['high_risk_f1'] for t in self.config['target_columns']])
                avg_hr_recall = np.mean([best_results[t]['high_risk_recall'] for t in self.config['target_columns']])
                
                comparison_data.append({
                    'strategy': f'Advanced_Sampling_{best_sampling}',
                    'avg_f1_macro': avg_f1,
                    'avg_recall_macro': avg_recall,
                    'avg_high_risk_f1': avg_hr_f1,
                    'avg_high_risk_recall': avg_hr_recall,
                    'combined_score': 0.2 * avg_f1 + 0.1 * avg_recall + 0.3 * avg_hr_f1 + 0.4 * avg_hr_recall
                })
        
        # Add ensemble and cascade results
        for strategy_name in ['Ensemble', 'Binary_Cascade']:
            if strategy_name.lower().replace('_', '') in [k.lower().replace('_', '') for k in all_strategies.keys()]:
                results = ensemble_results if strategy_name == 'Ensemble' else cascade_results
                
                avg_f1 = np.mean([results[t]['f1_macro'] for t in self.config['target_columns']])
                avg_recall = np.mean([results[t]['recall_macro'] for t in self.config['target_columns']])
                avg_hr_f1 = np.mean([results[t]['high_risk_f1'] for t in self.config['target_columns']])
                avg_hr_recall = np.mean([results[t]['high_risk_recall'] for t in self.config['target_columns']])
                
                comparison_data.append({
                    'strategy': strategy_name,
                    'avg_f1_macro': avg_f1,
                    'avg_recall_macro': avg_recall,
                    'avg_high_risk_f1': avg_hr_f1,
                    'avg_high_risk_recall': avg_hr_recall,
                    'combined_score': 0.2 * avg_f1 + 0.1 * avg_recall + 0.3 * avg_hr_f1 + 0.4 * avg_hr_recall
                })
        
        # Sort by combined score (weighted toward high-risk detection)
        comparison_data.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"\n📊 ADVANCED STRATEGIES COMPARISON:")
        print("-" * 60)
        
        for i, data in enumerate(comparison_data):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"\n{rank_emoji} {data['strategy'].upper()}:")
            print(f"   • F1-Macro: {data['avg_f1_macro']:.4f}")
            print(f"   • Recall: {data['avg_recall_macro']:.4f}")
            print(f"   • High-Risk F1: {data['avg_high_risk_f1']:.4f}")
            print(f"   • High-Risk Recall: {data['avg_high_risk_recall']:.1%} ⭐")
            print(f"   • Combined Score: {data['combined_score']:.4f}")
        
        # Select best strategy
        if comparison_data:
            self.best_approach = comparison_data[0]['strategy']
            best_hr_recall = comparison_data[0]['avg_high_risk_recall']
            
            print(f"\n🏆 BEST STRATEGY: {self.best_approach.upper()}")
            print(f"🎯 HIGH-RISK RECALL: {best_hr_recall:.1%}")
            
            # Compare with Step 5-1 results
            step5_1_hr_recall = 0.056  # From Step 5-1 results
            improvement = (best_hr_recall / step5_1_hr_recall - 1) * 100
            print(f"📈 IMPROVEMENT: {improvement:+.0f}% vs Step 5-1")
        
        # Store results
        self.results = {
            'strategies': {
                'sampling': sampling_results,
                'ensemble': ensemble_results,
                'cascade': cascade_results
            },
            'comparison': comparison_data,
            'best_approach': self.best_approach if hasattr(self, 'best_approach') else None
        }
        
        return comparison_data
    
    def create_visualizations(self):
        """Create visualization comparing all strategies"""
        print(f"\n6️⃣ CREATING VISUALIZATIONS")
        print("-" * 40)
        
        if not self.results['comparison']:
            print("   ❌ No results to visualize")
            return
        
        comparison_data = self.results['comparison']
        
        # High-risk recall comparison (most important metric)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Advanced Strategies: High-Risk Detection Performance', fontsize=16, fontweight='bold')
        
        strategies = [d['strategy'] for d in comparison_data]
        hr_recalls = [d['avg_high_risk_recall'] * 100 for d in comparison_data]  # Convert to percentage
        colors = ['#E74C3C', '#3498DB', '#2ECC71'][:len(strategies)]
        
        # High-risk recall bar chart
        bars = axes[0].bar(strategies, hr_recalls, color=colors)
        axes[0].set_title('High-Risk Recall Comparison', fontweight='bold')
        axes[0].set_ylabel('High-Risk Recall (%)')
        axes[0].set_xticklabels(strategies, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, recall in zip(bars, hr_recalls):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{recall:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add baseline reference line
        axes[0].axhline(y=5.6, color='red', linestyle='--', alpha=0.7, label='Step 5-1 Baseline (5.6%)')
        axes[0].legend()
        axes[0].set_ylim(0, max(hr_recalls) * 1.2)
        
        # Overall performance radar-like comparison
        metrics = ['F1-Macro', 'Recall', 'HR-F1', 'HR-Recall']
        
        for i, data in enumerate(comparison_data):
            values = [
                data['avg_f1_macro'] * 100,
                data['avg_recall_macro'] * 100,
                data['avg_high_risk_f1'] * 100,
                data['avg_high_risk_recall'] * 100
            ]
            axes[1].plot(metrics, values, marker='o', linewidth=2, 
                        label=data['strategy'], color=colors[i])
        
        axes[1].set_title('Overall Performance Comparison', fontweight='bold')
        axes[1].set_ylabel('Score (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/advanced_strategies_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("   ✅ Strategy comparison plot saved")
        
        plt.close('all')
    
    def save_results(self):
        """Save comprehensive results"""
        print(f"\n7️⃣ SAVING RESULTS")
        print("-" * 40)
        
        # Prepare results for JSON
        results_for_json = {
            'metadata': {
                'execution_date': datetime.now().isoformat(),
                'best_approach': self.best_approach,
                'goal': 'Achieve 15-25% high-risk recall',
                'baseline_hr_recall': 0.056  # From Step 5-1
            },
            'comparison_summary': self.results['comparison'],
            'detailed_results': self.results['strategies']
        }
        
        # Save results
        with open(f'{self.results_dir}/advanced_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        print("✅ Results saved successfully")
        print(f"   📁 Results: {self.results_dir}/advanced_results.json")
        print(f"   📁 Plots: {self.results_dir}/plots/")
    
    def run_complete_pipeline(self):
        """Run the complete advanced pipeline"""
        try:
            # Step 1: Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # Step 2: Advanced Sampling Strategy
            sampling_results = self.advanced_sampling_strategy(X_train, X_test, y_train, y_test)
            
            # Step 3: Ensemble Strategy
            ensemble_results = self.ensemble_strategy(X_train, X_test, y_train, y_test)
            
            # Step 4: Binary Cascade Strategy
            cascade_results = self.binary_cascade_strategy(X_train, X_test, y_train, y_test)
            
            # Step 5: Compare all strategies
            self.compare_all_strategies(sampling_results, ensemble_results, cascade_results)
            
            # Step 6: Create visualizations
            self.create_visualizations()
            
            # Step 7: Save results
            self.save_results()
            
            if hasattr(self, 'best_approach') and self.best_approach:
                print(f"\n🎉 ADVANCED PIPELINE COMPLETED!")
                print(f"   🏆 Best Strategy: {self.best_approach.upper()}")
                print(f"   🎯 Goal: Achieve 15-25% high-risk recall")
                print(f"   📈 Check results for improvement vs Step 5-1 baseline")
            else:
                print(f"\n⚠️  Pipeline completed with mixed results")
                print(f"   📊 Check detailed results for analysis")
            
        except Exception as e:
            print(f"\n❌ ERROR in Advanced Pipeline: {str(e)}")
            raise


def main():
    """Main execution function"""
    model = AdvancedImbalanceModel()
    model.run_complete_pipeline()


if __name__ == "__main__":
    main()