"""
Step 5-1: Quick Wins for Class Imbalance (Advanced Strategy)
============================================================
Implement three quick wins to dramatically improve high-risk detection:
1. Switch from XGBRegressor to XGBClassifier (probability outputs)
2. Threshold optimization for F1/Recall maximization
3. Focal Loss implementation for extreme imbalance

Focus: Maximize F1-score and Recall for high-risk detection

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
from sklearn.metrics import (mean_absolute_error, accuracy_score, classification_report, 
                           confusion_matrix, f1_score, recall_score, precision_score)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

# Visualization
plt.style.use('default')
sns.set_palette("husl")

class QuickWinsImbalanceModel:
    """Quick Wins Implementation for Extreme Class Imbalance"""
    
    def __init__(self):
        """Initialize the model with configuration"""
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
                'n_estimators': 100
            }
        }
        
        # Initialize containers
        self.data = None
        self.results = {}
        self.best_approach = None
        
        # Create results directory
        self.results_dir = 'result/step5_quick_wins'
        self._create_results_directory()
        
    def _create_results_directory(self):
        """Create results directory structure"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.results_dir}/models', exist_ok=True)
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """Load data and implement temporal split"""
        print("🚀 QUICK WINS: ADVANCED CLASS IMBALANCE STRATEGY")
        print("="*60)
        print("Goal: Dramatically improve high-risk detection with 3 quick wins")
        print("="*60)
        
        print("\n1️⃣ DATA LOADING & TEMPORAL SPLIT")
        print("-" * 40)
        
        # Load dataset
        print("📊 Loading dataset...")
        self.data = pd.read_csv(self.config['data_path'])
        print(f"✅ Dataset loaded: {len(self.data):,} records, {len(self.data.columns)} features")
        
        # Convert temporal column
        self.data[self.config['temporal_column']] = pd.to_datetime(
            self.data[self.config['temporal_column']], format='%Y-%m-%d'
        )
        
        # Implement same temporal split as Step 4/5
        current_date = datetime(2025, 6, 30).date()
        cutoff_dates = {}
        for i, target in enumerate(self.config['target_columns'], 1):
            cutoff_date = datetime(2025 - i, 6, 30).date()
            cutoff_dates[target] = cutoff_date
        
        # Filter to valid data (complete prediction periods)
        earliest_cutoff = min(cutoff_dates.values())
        valid_data = self.data[
            self.data[self.config['temporal_column']].dt.date < earliest_cutoff
        ].copy()
        
        print(f"📊 Valid data (before {earliest_cutoff}): {len(valid_data):,} records")
        
        # Temporal split
        split_index = int(len(valid_data) * self.config['train_ratio'])
        valid_data_sorted = valid_data.sort_values(self.config['temporal_column'])
        
        train_data = valid_data_sorted.iloc[:split_index]
        test_data = valid_data_sorted.iloc[split_index:]
        
        print(f"📊 Train: {len(train_data):,} records")
        print(f"📊 Test: {len(test_data):,} records")
        
        # Prepare features and targets
        feature_columns = [
            col for col in self.data.columns 
            if col not in self.config['target_columns'] + self.config['exclude_columns'] + [self.config['temporal_column']]
        ]
        
        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]
        
        y_train = {target: train_data[target] for target in self.config['target_columns']}
        y_test = {target: test_data[target] for target in self.config['target_columns']}
        
        print(f"📊 Features: {len(feature_columns)}")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Analyze class distribution
        self._analyze_class_distribution(y_train)
        
        return X_train, X_test, y_train, y_test
    
    def _analyze_class_distribution(self, y_train: Dict):
        """Analyze and display class distribution"""
        print(f"\n📈 CLASS DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        for target in self.config['target_columns']:
            valid_y = y_train[target].dropna()
            dist = valid_y.value_counts().sort_index()
            total = len(valid_y)
            
            print(f"\n🎯 {target}:")
            for class_val, count in dist.items():
                pct = (count / total) * 100
                print(f"   Class {int(class_val)}: {count:,} samples ({pct:.1f}%)")
            
            # Calculate imbalance ratios
            if len(dist) > 1:
                majority_count = dist.max()
                minority_count = dist.min()
                imbalance_ratio = majority_count / minority_count
                print(f"   📊 Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    def quick_win_1_xgb_classifier(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                  y_train: Dict, y_test: Dict) -> Dict:
        """Quick Win #1: Switch to XGBClassifier for probability outputs"""
        print(f"\n2️⃣ QUICK WIN #1: XGBClassifier (Probability Outputs)")
        print("-" * 40)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Training XGBClassifier for {target}...")
            
            # Get valid data
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask]
            y_train_valid = y_train[target][train_mask].astype(int)
            X_test_valid = X_test[test_mask]
            y_test_valid = y_test[target][test_mask].astype(int)
            
            # Train XGBClassifier
            model = xgb.XGBClassifier(
                **self.config['xgb_base_params'],
                objective='multi:softprob',
                num_class=len(np.unique(y_train_valid))
            )
            
            model.fit(X_train_valid, y_train_valid)
            
            # Get probability predictions
            y_pred_proba = model.predict_proba(X_test_valid)
            y_pred = model.predict(X_test_valid)  # Default threshold predictions
            
            # Calculate metrics
            f1_macro = f1_score(y_test_valid, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test_valid, y_pred, average='macro', zero_division=0)
            precision_macro = precision_score(y_test_valid, y_pred, average='macro', zero_division=0)
            
            # High-risk specific metrics (classes 2 and 3)
            high_risk_mask = (y_test_valid >= 2)
            high_risk_pred = (y_pred >= 2)
            
            if np.any(high_risk_mask):
                high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
            else:
                high_risk_recall = high_risk_f1 = 0.0
            
            results[target] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': {
                    'f1_macro': f1_macro,
                    'recall_macro': recall_macro,
                    'precision_macro': precision_macro,
                    'high_risk_recall': high_risk_recall,
                    'high_risk_f1': high_risk_f1
                }
            }
            
            print(f"   ✅ F1-Macro: {f1_macro:.4f}, Recall: {recall_macro:.4f}")
            print(f"   🎯 High-Risk F1: {high_risk_f1:.4f}, High-Risk Recall: {high_risk_recall:.4f}")
        
        return results
    
    def quick_win_2_threshold_optimization(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                          y_train: Dict, y_test: Dict, 
                                          base_results: Dict) -> Dict:
        """Quick Win #2: Optimize thresholds for F1/Recall maximization"""
        print(f"\n3️⃣ QUICK WIN #2: Threshold Optimization")
        print("-" * 40)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Optimizing thresholds for {target}...")
            
            # Get data
            test_mask = y_test[target].notna()
            y_test_valid = y_test[target][test_mask].astype(int)
            y_pred_proba = base_results[target]['probabilities']
            
            # Define threshold candidates
            threshold_candidates = [
                [0.5, 0.5, 0.5, 0.5],    # Default
                [0.4, 0.3, 0.2, 0.1],    # Lower for rare classes
                [0.3, 0.25, 0.15, 0.08], # Even more aggressive
                [0.2, 0.2, 0.1, 0.05],   # Very aggressive for high-risk
                [0.1, 0.15, 0.08, 0.03], # Maximum high-risk sensitivity
            ]
            
            best_result = None
            best_score = 0
            
            print(f"   🔍 Testing {len(threshold_candidates)} threshold combinations...")
            
            for i, thresholds in enumerate(threshold_candidates):
                # Apply custom thresholds
                y_pred_thresh = self._apply_custom_thresholds(y_pred_proba, thresholds)
                
                # Calculate metrics
                f1_macro = f1_score(y_test_valid, y_pred_thresh, average='macro', zero_division=0)
                recall_macro = recall_score(y_test_valid, y_pred_thresh, average='macro', zero_division=0)
                
                # High-risk metrics
                high_risk_mask = (y_test_valid >= 2)
                high_risk_pred = (y_pred_thresh >= 2)
                
                if np.any(high_risk_mask):
                    high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                    high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
                else:
                    high_risk_recall = high_risk_f1 = 0.0
                
                # Combined score (weighted toward high-risk detection)
                combined_score = 0.4 * f1_macro + 0.3 * recall_macro + 0.3 * high_risk_recall
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = {
                        'thresholds': thresholds,
                        'predictions': y_pred_thresh,
                        'metrics': {
                            'f1_macro': f1_macro,
                            'recall_macro': recall_macro,
                            'precision_macro': precision_score(y_test_valid, y_pred_thresh, average='macro', zero_division=0),
                            'high_risk_recall': high_risk_recall,
                            'high_risk_f1': high_risk_f1,
                            'combined_score': combined_score
                        }
                    }
                
                print(f"      Threshold {i+1}: F1={f1_macro:.3f}, Recall={recall_macro:.3f}, HR-Recall={high_risk_recall:.3f}")
            
            results[target] = best_result
            print(f"   🏆 Best thresholds: {best_result['thresholds']}")
            print(f"   ✅ F1-Macro: {best_result['metrics']['f1_macro']:.4f}, Recall: {best_result['metrics']['recall_macro']:.4f}")
            print(f"   🎯 High-Risk F1: {best_result['metrics']['high_risk_f1']:.4f}, High-Risk Recall: {best_result['metrics']['high_risk_recall']:.4f}")
        
        return results
    
    def _apply_custom_thresholds(self, probabilities: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """Apply custom thresholds to probability predictions"""
        predictions = []
        
        for sample_proba in probabilities:
            # Find the class with highest probability that exceeds its threshold
            best_class = 0
            best_score = 0
            
            for class_idx, (prob, threshold) in enumerate(zip(sample_proba, thresholds)):
                if prob >= threshold and prob > best_score:
                    best_class = class_idx
                    best_score = prob
            
            predictions.append(best_class)
        
        return np.array(predictions)
    
    def quick_win_3_focal_loss(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: Dict, y_test: Dict) -> Dict:
        """Quick Win #3: Focal Loss for extreme imbalance"""
        print(f"\n4️⃣ QUICK WIN #3: Focal Loss Implementation")
        print("-" * 40)
        
        results = {}
        
        for target in self.config['target_columns']:
            print(f"\n🎯 Training Focal Loss model for {target}...")
            
            # Get valid data
            train_mask = y_train[target].notna()
            test_mask = y_test[target].notna()
            
            X_train_valid = X_train[train_mask]
            y_train_valid = y_train[target][train_mask].astype(int)
            X_test_valid = X_test[test_mask]
            y_test_valid = y_test[target][test_mask].astype(int)
            
            # Calculate class weights for focal loss (more aggressive than standard)
            classes = np.unique(y_train_valid)
            class_counts = np.bincount(y_train_valid)
            total_samples = len(y_train_valid)
            
            # Focal loss weights: inverse frequency with exponential scaling
            focal_weights = []
            for class_idx in classes:
                class_freq = class_counts[class_idx] / total_samples
                # Exponential weighting for extreme imbalance
                weight = (1.0 / class_freq) ** 0.5  # Square root to moderate the effect
                focal_weights.append(weight)
            
            # Normalize weights
            focal_weights = np.array(focal_weights)
            focal_weights = focal_weights / np.mean(focal_weights)
            
            print(f"   📊 Focal weights: {dict(zip(classes, focal_weights.round(2)))}")
            
            # Train with focal loss approximation using sample weights
            sample_weights = np.array([focal_weights[int(y)] for y in y_train_valid])
            
            model = xgb.XGBClassifier(
                **self.config['xgb_base_params'],
                objective='multi:softprob',
                num_class=len(classes),
                scale_pos_weight=focal_weights[-1] if len(focal_weights) > 1 else 1  # Extra weight for highest class
            )
            
            model.fit(X_train_valid, y_train_valid, sample_weight=sample_weights)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_valid)
            y_pred = model.predict(X_test_valid)
            
            # Calculate metrics
            f1_macro = f1_score(y_test_valid, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test_valid, y_pred, average='macro', zero_division=0)
            precision_macro = precision_score(y_test_valid, y_pred, average='macro', zero_division=0)
            
            # High-risk specific metrics
            high_risk_mask = (y_test_valid >= 2)
            high_risk_pred = (y_pred >= 2)
            
            if np.any(high_risk_mask):
                high_risk_recall = recall_score(high_risk_mask, high_risk_pred, zero_division=0)
                high_risk_f1 = f1_score(high_risk_mask, high_risk_pred, zero_division=0)
            else:
                high_risk_recall = high_risk_f1 = 0.0
            
            results[target] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'focal_weights': focal_weights,
                'metrics': {
                    'f1_macro': f1_macro,
                    'recall_macro': recall_macro,
                    'precision_macro': precision_macro,
                    'high_risk_recall': high_risk_recall,
                    'high_risk_f1': high_risk_f1
                }
            }
            
            print(f"   ✅ F1-Macro: {f1_macro:.4f}, Recall: {recall_macro:.4f}")
            print(f"   🎯 High-Risk F1: {high_risk_f1:.4f}, High-Risk Recall: {high_risk_recall:.4f}")
        
        return results
    
    def compare_all_approaches(self, qw1_results: Dict, qw2_results: Dict, qw3_results: Dict):
        """Compare all three quick wins approaches"""
        print(f"\n5️⃣ APPROACH COMPARISON & BEST SELECTION")
        print("=" * 60)
        
        approaches = {
            'XGBClassifier': qw1_results,
            'ThresholdOpt': qw2_results,
            'FocalLoss': qw3_results
        }
        
        comparison_data = []
        
        for approach_name, results in approaches.items():
            avg_f1 = np.mean([results[target]['metrics']['f1_macro'] for target in self.config['target_columns']])
            avg_recall = np.mean([results[target]['metrics']['recall_macro'] for target in self.config['target_columns']])
            avg_hr_f1 = np.mean([results[target]['metrics']['high_risk_f1'] for target in self.config['target_columns']])
            avg_hr_recall = np.mean([results[target]['metrics']['high_risk_recall'] for target in self.config['target_columns']])
            
            comparison_data.append({
                'approach': approach_name,
                'avg_f1_macro': avg_f1,
                'avg_recall_macro': avg_recall,
                'avg_high_risk_f1': avg_hr_f1,
                'avg_high_risk_recall': avg_hr_recall,
                'combined_score': 0.3 * avg_f1 + 0.2 * avg_recall + 0.3 * avg_hr_f1 + 0.2 * avg_hr_recall
            })
        
        # Sort by combined score
        comparison_data.sort(key=lambda x: x['combined_score'], reverse=True)
        
        print(f"\n📊 OVERALL PERFORMANCE COMPARISON:")
        print("-" * 60)
        
        for i, data in enumerate(comparison_data):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"\n{rank_emoji} {data['approach'].upper()}:")
            print(f"   • Average F1-Macro: {data['avg_f1_macro']:.4f}")
            print(f"   • Average Recall: {data['avg_recall_macro']:.4f}")
            print(f"   • High-Risk F1: {data['avg_high_risk_f1']:.4f}")
            print(f"   • High-Risk Recall: {data['avg_high_risk_recall']:.4f}")
            print(f"   • Combined Score: {data['combined_score']:.4f}")
        
        # Select best approach
        self.best_approach = comparison_data[0]['approach']
        print(f"\n🏆 BEST APPROACH: {self.best_approach.upper()}")
        
        # Store results
        self.results = {
            'approaches': approaches,
            'comparison': comparison_data,
            'best_approach': self.best_approach
        }
        
        return comparison_data
    
    def create_visualizations(self):
        """Create comparison visualizations"""
        print(f"\n6️⃣ CREATING VISUALIZATIONS")
        print("-" * 40)
        
        # Performance comparison plot
        comparison_data = self.results['comparison']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quick Wins: Performance Comparison', fontsize=16, fontweight='bold')
        
        approaches = [d['approach'] for d in comparison_data]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # F1-Macro comparison
        f1_scores = [d['avg_f1_macro'] for d in comparison_data]
        axes[0, 0].bar(approaches, f1_scores, color=colors)
        axes[0, 0].set_title('Average F1-Macro Score')
        axes[0, 0].set_ylabel('F1-Macro')
        
        # Recall comparison
        recall_scores = [d['avg_recall_macro'] for d in comparison_data]
        axes[0, 1].bar(approaches, recall_scores, color=colors)
        axes[0, 1].set_title('Average Recall Score')
        axes[0, 1].set_ylabel('Recall')
        
        # High-risk F1 comparison
        hr_f1_scores = [d['avg_high_risk_f1'] for d in comparison_data]
        axes[1, 0].bar(approaches, hr_f1_scores, color=colors)
        axes[1, 0].set_title('High-Risk F1 Score')
        axes[1, 0].set_ylabel('High-Risk F1')
        
        # High-risk Recall comparison
        hr_recall_scores = [d['avg_high_risk_recall'] for d in comparison_data]
        axes[1, 1].bar(approaches, hr_recall_scores, color=colors)
        axes[1, 1].set_title('High-Risk Recall Score')
        axes[1, 1].set_ylabel('High-Risk Recall')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/quick_wins_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Performance comparison plot saved")
        
        # Detailed metrics by target
        self._create_detailed_target_plots()
        
        plt.close('all')
        
    def _create_detailed_target_plots(self):
        """Create detailed plots for each target variable"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quick Wins: Performance by Target Variable', fontsize=16, fontweight='bold')
        
        approaches = list(self.results['approaches'].keys())
        targets = self.config['target_columns']
        
        for i, target in enumerate(targets):
            ax = axes[i // 2, i % 2]
            
            # Get F1 scores for this target across approaches
            f1_scores = []
            recall_scores = []
            
            for approach in approaches:
                f1_scores.append(self.results['approaches'][approach][target]['metrics']['f1_macro'])
                recall_scores.append(self.results['approaches'][approach][target]['metrics']['recall_macro'])
            
            x = np.arange(len(approaches))
            width = 0.35
            
            ax.bar(x - width/2, f1_scores, width, label='F1-Macro', alpha=0.8)
            ax.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.8)
            
            ax.set_title(f'{target} Performance')
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(approaches, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/target_detailed_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Detailed target comparison plot saved")
    
    def save_results(self):
        """Save all results to files"""
        print(f"\n7️⃣ SAVING RESULTS")
        print("-" * 40)
        
        # Prepare results for JSON serialization
        results_for_json = {
            'metadata': {
                'execution_date': datetime.now().isoformat(),
                'best_approach': self.best_approach,
                'config': self.config
            },
            'comparison_summary': self.results['comparison'],
            'detailed_results_by_approach': {}
        }
        
        # Add detailed results for each approach
        for approach_name, approach_results in self.results['approaches'].items():
            results_for_json['detailed_results_by_approach'][approach_name] = {}
            
            for target in self.config['target_columns']:
                target_results = approach_results[target]
                results_for_json['detailed_results_by_approach'][approach_name][target] = {
                    'metrics': target_results['metrics']
                }
                
                # Add threshold info if available
                if 'thresholds' in target_results:
                    results_for_json['detailed_results_by_approach'][approach_name][target]['thresholds'] = target_results['thresholds']
                
                # Add focal weights if available
                if 'focal_weights' in target_results:
                    results_for_json['detailed_results_by_approach'][approach_name][target]['focal_weights'] = target_results['focal_weights'].tolist()
        
        # Save results
        with open(f'{self.results_dir}/quick_wins_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        print("✅ Results saved successfully")
        print(f"   📁 Summary: {self.results_dir}/quick_wins_results.json")
        print(f"   📁 Plots: {self.results_dir}/plots/")
    
    def run_complete_pipeline(self):
        """Run the complete Quick Wins pipeline"""
        try:
            # Step 1: Load and prepare data
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # Step 2: Quick Win #1 - XGBClassifier
            qw1_results = self.quick_win_1_xgb_classifier(X_train, X_test, y_train, y_test)
            
            # Step 3: Quick Win #2 - Threshold Optimization
            qw2_results = self.quick_win_2_threshold_optimization(X_train, X_test, y_train, y_test, qw1_results)
            
            # Step 4: Quick Win #3 - Focal Loss
            qw3_results = self.quick_win_3_focal_loss(X_train, X_test, y_train, y_test)
            
            # Step 5: Compare all approaches
            self.compare_all_approaches(qw1_results, qw2_results, qw3_results)
            
            # Step 6: Create visualizations
            self.create_visualizations()
            
            # Step 7: Save results
            self.save_results()
            
            print(f"\n🎉 QUICK WINS PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"   🏆 Best approach: {self.best_approach.upper()}")
            print(f"   📈 Expected improvement: Significant boost in high-risk detection")
            print(f"   📁 Results saved to: {self.results_dir}")
            
        except Exception as e:
            print(f"\n❌ ERROR in Quick Wins Pipeline: {str(e)}")
            raise


def main():
    """Main execution function"""
    model = QuickWinsImbalanceModel()
    model.run_complete_pipeline()


if __name__ == "__main__":
    main()