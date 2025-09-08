"""
XGBoost Risk Prediction Model - Step 1: Multi-Model Evaluation
============================================================

Step 1 Implementation (EVALUATION FOCUS):
- Load trained models and predictions from step1_train.py
- Comprehensive evaluation of XGBoost, MLP, and RandomForest models
- Deep analysis: confusion matrices, error patterns, feature importance
- Business-focused performance interpretation
- Clean separation: Evaluation only, training handled by step1_train.py

Design Focus:
- Load standardized outputs from training step
- Comprehensive evaluation metrics for multi-class classification
- Business-meaningful insights and recommendations
- Korean font support for visualizations
- Performance comparison across algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    precision_recall_fscore_support, cohen_kappa_score, 
    balanced_accuracy_score, classification_report
)
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib
import json
import os
import warnings
import platform
import matplotlib.font_manager as fm
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

# Set up Korean font and style
setup_korean_font()
plt.style.use('default')
sns.set_palette("husl")


class MultiModelEvaluator:
    """
    Step 1: Multi-Model Evaluation Framework
    
    Loads results from step1_train.py and performs comprehensive evaluation:
    - Performance metrics comparison
    - Confusion matrix analysis
    - Error pattern analysis
    - Feature importance analysis
    - Business impact assessment
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator
        
        Args:
            config: Dictionary containing evaluation configuration
        """
        self.config = config
        self.training_dir = config.get('training_results_dir', '../results/step1_baseline')
        self.results_dir = self._create_results_directory()
        
        # Data structures for loaded results
        self.metadata = None
        self.predictions_df = None
        self.models = {}
        self.feature_columns = None
        self.evaluation_results = {}
        
        # Risk level definitions
        self.risk_levels = {
            0: "위험없음 (No Risk)",
            1: "낮은위험 (Low Risk)", 
            2: "중간위험 (Medium Risk)",
            3: "높은위험 (High Risk)"
        }
        
        print("🔍 Multi-Model Evaluator Initialized")
        print(f"📁 Loading results from: {self.training_dir}")
        print(f"📁 Evaluation outputs to: {self.results_dir}")
    
    def _create_results_directory(self) -> str:
        """Create results directory for evaluation outputs"""
        results_dir = "../results/step1_evaluation"
        
        os.makedirs(results_dir, exist_ok=True)
        
        subdirs = ['visualizations', 'metrics', 'analysis']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir
    
    def _save_plot(self, filename: str, dpi: int = 300) -> str:
        """Save current plot to visualizations directory"""
        save_path = os.path.join(self.results_dir, 'visualizations', f"{filename}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  💾 Saved: {filename}.png")
        return save_path
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    
    def load_training_results(self):
        """Load results from step1_train.py"""
        print("\n" + "="*60)
        print("1️⃣ LOADING TRAINING RESULTS")
        print("="*60)
        
        # Load metadata
        metadata_path = os.path.join(self.training_dir, 'metadata', 'training_metadata.json')
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata: {os.path.basename(metadata_path)}")
            print(f"   • Training date: {self.metadata['training_timestamp']}")
            print(f"   • Algorithms: {self.metadata['algorithms_trained']}")
            print(f"   • Total models: {sum(self.metadata['model_counts'].values())}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            raise
        
        # Load predictions
        predictions_path = os.path.join(self.training_dir, 'predictions', 'all_predictions.csv')
        try:
            self.predictions_df = pd.read_csv(predictions_path)
            print(f"✅ Loaded predictions: {len(self.predictions_df):,} records")
            
            # Verify prediction structure
            expected_cols = ['algorithm', 'target', 'test_index', 'predicted_class', 'actual_class']
            missing_cols = [col for col in expected_cols if col not in self.predictions_df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in predictions: {missing_cols}")
            
            print(f"   • Algorithms in predictions: {sorted(self.predictions_df['algorithm'].unique())}")
            print(f"   • Targets in predictions: {sorted(self.predictions_df['target'].unique())}")
            
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            raise
        
        # Load trained models (for feature importance analysis)
        print(f"\n🤖 Loading trained models...")
        models_dir = os.path.join(self.training_dir, 'models')
        
        for alg_name in self.metadata['algorithms_trained']:
            self.models[alg_name] = {}
            for target in self.metadata['target_columns']:
                try:
                    if alg_name == 'xgboost':
                        model_path = os.path.join(models_dir, f'{target}_{alg_name}_model.json')
                        if os.path.exists(model_path):
                            # Use Booster directly to avoid sklearn compatibility issues
                            booster = xgb.Booster()
                            booster.load_model(model_path)
                            self.models[alg_name][target] = booster
                    else:
                        model_path = os.path.join(models_dir, f'{target}_{alg_name}_model.joblib')
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            self.models[alg_name][target] = model
                except Exception as e:
                    print(f"   ⚠️  Could not load {alg_name} model for {target}: {e}")
        
        # Store feature columns
        self.feature_columns = self.metadata['feature_columns']
        
        print(f"✅ Training results loaded successfully")
        return True
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        print("\n" + "="*60)
        print("2️⃣ COMPREHENSIVE METRICS CALCULATION")
        print("="*60)
        
        evaluation_results = {}
        
        for algorithm in self.metadata['algorithms_trained']:
            print(f"\n📊 Evaluating {algorithm.upper()}...")
            evaluation_results[algorithm] = {}
            
            alg_predictions = self.predictions_df[self.predictions_df['algorithm'] == algorithm]
            
            for target in self.metadata['target_columns']:
                target_predictions = alg_predictions[alg_predictions['target'] == target]
                
                if len(target_predictions) == 0:
                    print(f"   ⚠️  No predictions for {target}")
                    continue
                
                y_true = target_predictions['actual_class'].values
                y_pred = target_predictions['predicted_class'].values
                
                # Basic classification metrics
                accuracy = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                kappa = cohen_kappa_score(y_true, y_pred)
                
                # Class-wise metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
                )
                
                # High-risk detection metrics (classes 2 and 3)
                high_risk_true = (y_true >= 2).astype(int)
                high_risk_pred = (y_pred >= 2).astype(int)
                
                high_risk_precision = precision_recall_fscore_support(
                    high_risk_true, high_risk_pred, average='binary', zero_division=0
                )[0]
                high_risk_recall = precision_recall_fscore_support(
                    high_risk_true, high_risk_pred, average='binary', zero_division=0
                )[1]
                high_risk_f1 = precision_recall_fscore_support(
                    high_risk_true, high_risk_pred, average='binary', zero_division=0
                )[2]
                
                # Class distribution analysis
                true_dist = pd.Series(y_true).value_counts().sort_index()
                pred_dist = pd.Series(y_pred).value_counts().sort_index()
                
                # Store results
                evaluation_results[algorithm][target] = {
                    'basic_metrics': {
                        'accuracy': float(accuracy),
                        'f1_macro': float(f1_macro),
                        'f1_weighted': float(f1_weighted),
                        'balanced_accuracy': float(balanced_acc),
                        'cohen_kappa': float(kappa)
                    },
                    'high_risk_metrics': {
                        'precision': float(high_risk_precision),
                        'recall': float(high_risk_recall),
                        'f1_score': float(high_risk_f1)
                    },
                    'class_wise_metrics': {
                        'precision': [float(p) for p in precision],
                        'recall': [float(r) for r in recall],
                        'f1_score': [float(f) for f in f1],
                        'support': [int(s) for s in support]
                    },
                    'distributions': {
                        'true_distribution': dict(true_dist),
                        'predicted_distribution': dict(pred_dist)
                    },
                    'sample_size': len(y_true)
                }
                
                print(f"   📈 {target}: Acc={accuracy:.4f}, F1={f1_macro:.4f}, HighRisk-Recall={high_risk_recall:.4f}")
        
        self.evaluation_results = evaluation_results
        
        # Save evaluation metrics
        self._save_evaluation_metrics()
        
        return evaluation_results
    
    def _save_evaluation_metrics(self):
        """Save detailed evaluation metrics to files"""
        print("\n💾 Saving evaluation metrics...")
        
        # Create summary table for CSV
        summary_data = []
        for algorithm, targets in self.evaluation_results.items():
            for target, metrics in targets.items():
                summary_data.append({
                    'Algorithm': algorithm.upper(),
                    'Target': target,
                    'Accuracy': metrics['basic_metrics']['accuracy'],
                    'F1_Macro': metrics['basic_metrics']['f1_macro'],
                    'F1_Weighted': metrics['basic_metrics']['f1_weighted'],
                    'Balanced_Accuracy': metrics['basic_metrics']['balanced_accuracy'],
                    'Cohen_Kappa': metrics['basic_metrics']['cohen_kappa'],
                    'HighRisk_Precision': metrics['high_risk_metrics']['precision'],
                    'HighRisk_Recall': metrics['high_risk_metrics']['recall'],
                    'HighRisk_F1': metrics['high_risk_metrics']['f1_score'],
                    'Sample_Size': metrics['sample_size']
                })
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, 'metrics', 'comprehensive_evaluation.csv')
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   📊 Summary saved: comprehensive_evaluation.csv")
        
        # Save detailed JSON with proper serialization
        json_path = os.path.join(self.results_dir, 'metrics', 'detailed_metrics.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        print(f"   📄 Details saved: detailed_metrics.json")
    
    def create_confusion_matrices(self):
        """Create comprehensive confusion matrix analysis"""
        print("\n" + "="*60)
        print("3️⃣ CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        algorithms = self.metadata['algorithms_trained']
        targets = self.metadata['target_columns']
        
        # Create confusion matrices for each algorithm
        for algorithm in algorithms:
            print(f"\n📊 Creating confusion matrices for {algorithm.upper()}...")
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            fig.suptitle(f'Confusion Matrix Analysis - {algorithm.upper()}', 
                        fontsize=16, fontweight='bold')
            
            alg_predictions = self.predictions_df[self.predictions_df['algorithm'] == algorithm]
            axes = axes.flatten()
            
            for i, target in enumerate(targets):
                if i >= 4:
                    break
                
                target_predictions = alg_predictions[alg_predictions['target'] == target]
                if len(target_predictions) == 0:
                    continue
                
                y_true = target_predictions['actual_class'].values
                y_pred = target_predictions['predicted_class'].values
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Create heatmap
                ax = axes[i]
                sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                           xticklabels=[self.risk_levels[j] for j in [0,1,2,3]],
                           yticklabels=[self.risk_levels[j] for j in [0,1,2,3]],
                           ax=ax)
                
                ax.set_title(f'{target} - Confusion Matrix (Normalized)', fontsize=12)
                ax.set_xlabel('Predicted Risk Level', fontsize=10)
                ax.set_ylabel('Actual Risk Level', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='y', rotation=0)
            
            plt.tight_layout()
            self._save_plot(f"01_confusion_matrices_{algorithm}")
    
    def analyze_error_patterns(self):
        """Analyze error patterns with business impact focus"""
        print("\n" + "="*60)
        print("4️⃣ ERROR PATTERN ANALYSIS")
        print("="*60)
        
        algorithms = self.metadata['algorithms_trained']
        
        # Create error analysis for each algorithm
        for algorithm in algorithms:
            print(f"\n🔍 Analyzing error patterns for {algorithm.upper()}...")
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'Error Pattern Analysis - {algorithm.upper()}', 
                        fontsize=16, fontweight='bold')
            
            alg_predictions = self.predictions_df[self.predictions_df['algorithm'] == algorithm]
            axes = axes.flatten()
            
            for i, target in enumerate(self.metadata['target_columns']):
                if i >= 4:
                    break
                
                target_predictions = alg_predictions[alg_predictions['target'] == target]
                if len(target_predictions) == 0:
                    continue
                
                y_true = target_predictions['actual_class'].values
                y_pred = target_predictions['predicted_class'].values
                errors = y_pred - y_true
                
                # Create error distribution plot
                ax = axes[i]
                error_counts = pd.Series(errors).value_counts().sort_index()
                
                colors = ['red' if x < 0 else 'orange' if x > 0 else 'green' for x in error_counts.index]
                ax.bar(error_counts.index, error_counts.values, color=colors, alpha=0.7)
                
                ax.set_title(f'{target} - Prediction Errors', fontsize=12)
                ax.set_xlabel('Error (Predicted - Actual)', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add error statistics
                underestimate = (errors < 0).sum()
                overestimate = (errors > 0).sum()
                perfect = (errors == 0).sum()
                
                ax.text(0.02, 0.98, 
                       f'Perfect: {perfect}\nUnder-estimate: {underestimate}\nOver-estimate: {overestimate}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._save_plot(f"02_error_patterns_{algorithm}")
        
        # Business impact analysis
        self._analyze_business_impact()
    
    def _analyze_business_impact(self):
        """Analyze business impact of prediction errors"""
        print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
        
        impact_analysis = {}
        
        for algorithm in self.metadata['algorithms_trained']:
            print(f"\n   🎯 {algorithm.upper()}:")
            impact_analysis[algorithm] = {}
            
            alg_predictions = self.predictions_df[self.predictions_df['algorithm'] == algorithm]
            
            for target in self.metadata['target_columns']:
                target_predictions = alg_predictions[alg_predictions['target'] == target]
                if len(target_predictions) == 0:
                    continue
                
                y_true = target_predictions['actual_class'].values
                y_pred = target_predictions['predicted_class'].values
                
                # High-risk detection analysis
                total_high_risk = (y_true >= 2).sum()
                missed_high_risk = ((y_true >= 2) & (y_pred < 2)).sum()
                false_high_risk = ((y_true < 2) & (y_pred >= 2)).sum()
                
                detection_rate = (1 - missed_high_risk/max(total_high_risk, 1)) * 100
                false_alarm_rate = false_high_risk / max((y_true < 2).sum(), 1) * 100
                
                impact_analysis[algorithm][target] = {
                    'total_high_risk': int(total_high_risk),
                    'missed_high_risk': int(missed_high_risk),
                    'false_high_risk': int(false_high_risk),
                    'detection_rate': float(detection_rate),
                    'false_alarm_rate': float(false_alarm_rate)
                }
                
                print(f"     • {target}:")
                print(f"       - High-risk cases: {total_high_risk}")
                print(f"       - Missed high-risk: {missed_high_risk} ({100-detection_rate:.1f}%)")
                print(f"       - False alarms: {false_high_risk} ({false_alarm_rate:.1f}%)")
                print(f"       - Detection rate: {detection_rate:.1f}%")
        
        # Save business impact analysis
        impact_path = os.path.join(self.results_dir, 'analysis', 'business_impact.json')
        with open(impact_path, 'w', encoding='utf-8') as f:
            json.dump(impact_analysis, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        print(f"\n   📄 Business impact analysis saved: business_impact.json")
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison across algorithms"""
        print("\n" + "="*60)
        print("5️⃣ PERFORMANCE COMPARISON")
        print("="*60)
        
        # Prepare comparison data
        comparison_data = []
        for algorithm, targets in self.evaluation_results.items():
            for target, metrics in targets.items():
                comparison_data.append({
                    'Algorithm': algorithm.upper(),
                    'Target': target,
                    'Accuracy': metrics['basic_metrics']['accuracy'],
                    'F1_Macro': metrics['basic_metrics']['f1_macro'],
                    'Balanced_Acc': metrics['basic_metrics']['balanced_accuracy'],
                    'Cohen_Kappa': metrics['basic_metrics']['cohen_kappa'],
                    'HighRisk_Recall': metrics['high_risk_metrics']['recall'],
                    'HighRisk_F1': metrics['high_risk_metrics']['f1_score']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comprehensive comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Multi-Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('Accuracy', 'Accuracy'),
            ('F1_Macro', 'F1-Score (Macro)'),
            ('Balanced_Acc', 'Balanced Accuracy'),
            ('Cohen_Kappa', 'Cohen\'s Kappa'),
            ('HighRisk_Recall', 'High-Risk Recall'),
            ('HighRisk_F1', 'High-Risk F1-Score')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            # Create grouped bar chart
            pivot_data = comparison_df.pivot(index='Target', columns='Algorithm', values=metric)
            pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
            
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Score', fontsize=10)
            ax.legend(title='Algorithm', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        self._save_plot("03_performance_comparison")
        
        # Print performance ranking
        self._print_performance_ranking(comparison_df)
    
    def _print_performance_ranking(self, comparison_df: pd.DataFrame):
        """Print performance ranking by different metrics"""
        print(f"\n🏆 PERFORMANCE RANKING:")
        
        # Overall ranking by F1-Macro
        overall_ranking = comparison_df.groupby('Algorithm')['F1_Macro'].mean().sort_values(ascending=False)
        print(f"\n   📊 Overall Ranking (by average F1-Macro):")
        for rank, (algorithm, score) in enumerate(overall_ranking.items(), 1):
            print(f"      {rank}. {algorithm}: {score:.4f}")
        
        # High-risk detection ranking
        highrisk_ranking = comparison_df.groupby('Algorithm')['HighRisk_Recall'].mean().sort_values(ascending=False)
        print(f"\n   🚨 High-Risk Detection Ranking (by average recall):")
        for rank, (algorithm, score) in enumerate(highrisk_ranking.items(), 1):
            print(f"      {rank}. {algorithm}: {score:.4f}")
        
        # Balanced accuracy ranking
        balanced_ranking = comparison_df.groupby('Algorithm')['Balanced_Acc'].mean().sort_values(ascending=False)
        print(f"\n   ⚖️  Balanced Performance Ranking (by balanced accuracy):")
        for rank, (algorithm, score) in enumerate(balanced_ranking.items(), 1):
            print(f"      {rank}. {algorithm}: {score:.4f}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance for models that support it"""
        print("\n" + "="*60)
        print("6️⃣ FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        for algorithm in ['xgboost', 'randomforest']:  # MLP doesn't have feature_importances_
            if algorithm not in self.models or not self.models[algorithm]:
                print(f"   ⚠️  No {algorithm} models available for feature importance")
                continue
            
            print(f"\n📊 Analyzing feature importance for {algorithm.upper()}...")
            
            importance_data = {}
            for target, model in self.models[algorithm].items():
                try:
                    if algorithm == 'xgboost' and hasattr(model, 'get_score'):
                        # For XGBoost Booster
                        importance_dict = model.get_score(importance_type='gain')
                        # Convert to array format matching feature_columns order
                        importance_array = np.zeros(len(self.feature_columns))
                        for i, feature in enumerate(self.feature_columns):
                            feature_key = f'f{i}'  # XGBoost uses f0, f1, f2... format
                            importance_array[i] = importance_dict.get(feature_key, 0.0)
                        importance_data[target] = importance_array
                    elif hasattr(model, 'feature_importances_'):
                        importance_data[target] = model.feature_importances_
                    elif isinstance(model, Pipeline) and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
                        importance_data[target] = model.named_steps['classifier'].feature_importances_
                except Exception as e:
                    print(f"      ⚠️  Could not extract importance for {target}: {e}")
            
            if not importance_data:
                print(f"      ❌ No feature importance data available for {algorithm}")
                continue
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame(importance_data, index=self.feature_columns)
            importance_df['avg_importance'] = importance_df.mean(axis=1)
            top_features = importance_df.nlargest(20, 'avg_importance')
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
            fig.suptitle(f'Feature Importance Analysis - {algorithm.upper()}', fontsize=16, fontweight='bold')
            
            # Average importance
            top_features['avg_importance'].plot(kind='barh', ax=ax1)
            ax1.set_title('Top 20 Features - Average Importance')
            ax1.set_xlabel('Importance Score')
            
            # Per-target importance
            target_cols = [col for col in top_features.columns if col != 'avg_importance']
            if target_cols:
                top_features[target_cols].plot(kind='barh', ax=ax2)
                ax2.set_title('Feature Importance by Target Variable')
                ax2.set_xlabel('Importance Score')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            self._save_plot(f"04_feature_importance_{algorithm}")
            
            # Print top features
            print(f"\n   🏆 TOP 10 MOST IMPORTANT FEATURES ({algorithm.upper()}):")
            for i, (feature, importance) in enumerate(top_features['avg_importance'].head(10).items(), 1):
                print(f"      {i:2d}. {feature}: {importance:.4f}")
            
            # Save feature importance data
            importance_path = os.path.join(self.results_dir, 'analysis', f'feature_importance_{algorithm}.csv')
            top_features.to_csv(importance_path, encoding='utf-8-sig')
            print(f"      📄 Feature importance saved: feature_importance_{algorithm}.csv")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("7️⃣ EVALUATION REPORT GENERATION")
        print("="*60)
        
        # Calculate summary statistics
        all_metrics = []
        for algorithm, targets in self.evaluation_results.items():
            for target, metrics in targets.items():
                all_metrics.append({
                    'algorithm': algorithm,
                    'target': target,
                    **metrics['basic_metrics'],
                    **metrics['high_risk_metrics']
                })
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Generate report
        report = {
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_metadata': self.metadata,
            'summary_statistics': {
                'total_predictions_evaluated': len(self.predictions_df),
                'algorithms_evaluated': self.metadata['algorithms_trained'],
                'targets_evaluated': self.metadata['target_columns']
            },
            'performance_summary': {
                'best_overall_algorithm': metrics_df.groupby('algorithm')['f1_macro'].mean().idxmax(),
                'best_highrisk_algorithm': metrics_df.groupby('algorithm')['recall'].mean().idxmax(),
                'average_metrics': {
                    'accuracy': float(metrics_df['accuracy'].mean()),
                    'f1_macro': float(metrics_df['f1_macro'].mean()),
                    'balanced_accuracy': float(metrics_df['balanced_accuracy'].mean()),
                    'high_risk_recall': float(metrics_df['recall'].mean())
                }
            },
            'recommendations': [
                f"Best overall performer: {metrics_df.groupby('algorithm')['f1_macro'].mean().idxmax()}",
                f"Best for high-risk detection: {metrics_df.groupby('algorithm')['recall'].mean().idxmax()}",
                "Consider ensemble methods combining strengths of different algorithms",
                "Focus on improving high-risk recall for business impact",
                "Investigate feature engineering to boost performance",
                "Consider temporal validation for realistic performance estimation"
            ]
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        print(f"✅ Comprehensive evaluation report saved: evaluation_report.json")
        
        # Print executive summary
        print(f"\n📋 EXECUTIVE SUMMARY:")
        print("=" * 60)
        print(f"📅 Evaluation Date: {report['evaluation_timestamp']}")
        print(f"🏆 Best Overall: {report['performance_summary']['best_overall_algorithm'].upper()}")
        print(f"🚨 Best High-Risk Detection: {report['performance_summary']['best_highrisk_algorithm'].upper()}")
        print(f"📊 Average Performance:")
        for metric, value in report['performance_summary']['average_metrics'].items():
            print(f"   • {metric.replace('_', ' ').title()}: {value:.4f}")
    
    def run_evaluation_pipeline(self):
        """Execute complete evaluation pipeline"""
        print("🔍 EXECUTING STEP 1: MULTI-MODEL EVALUATION PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load training results
            self.load_training_results()
            
            # Step 2: Calculate comprehensive metrics
            self.calculate_comprehensive_metrics()
            
            # Step 3: Create confusion matrices
            self.create_confusion_matrices()
            
            # Step 4: Analyze error patterns
            self.analyze_error_patterns()
            
            # Step 5: Create performance comparison
            self.create_performance_comparison()
            
            # Step 6: Analyze feature importance
            self.analyze_feature_importance()
            
            # Step 7: Generate evaluation report
            self.generate_evaluation_report()
            
            print("\n🎉 STEP 1 EVALUATION COMPLETED SUCCESSFULLY!")
            print("✅ Comprehensive multi-model evaluation completed")
            print("✅ Performance comparison established")
            print("✅ Business insights generated")
            print("✅ Ready for next phases")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 1 EVALUATION FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of evaluation outputs"""
        print(f"\n📁 EVALUATION RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Visualizations:")
        for alg in self.metadata['algorithms_trained']:
            print(f"      • 01_confusion_matrices_{alg}.png")
            print(f"      • 02_error_patterns_{alg}.png")
        print(f"      • 03_performance_comparison.png")
        for alg in ['xgboost', 'randomforest']:
            print(f"      • 04_feature_importance_{alg}.png")
        print(f"   📈 Metrics & Analysis:")
        print(f"      • comprehensive_evaluation.csv")
        print(f"      • detailed_metrics.json")
        print(f"      • business_impact.json")
        print(f"      • feature_importance_*.csv")
        print(f"   📄 Report:")
        print(f"      • evaluation_report.json")
        print(f"\n💡 Use these insights to guide subsequent modeling phases!")


def get_evaluation_config():
    """Configuration for Step 1 evaluation"""
    return {
        'training_results_dir': '../results/step1_baseline'
    }


# Main execution
if __name__ == "__main__":
    print("🔍 Starting Multi-Model Evaluation - Step 1")
    print("="*60)
    
    # Get configuration
    config = get_evaluation_config()
    
    # Create and run evaluator
    evaluator = MultiModelEvaluator(config)
    evaluator.run_evaluation_pipeline()
    
    print("\n🏁 Step 1 Evaluation execution completed!")
    print("✅ Multi-model baseline evaluation completed successfully")
    print("Ready to proceed to Step 2: Advanced modeling phases!")
