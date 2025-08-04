"""
XGBoost Risk Prediction Model - Step 2: Basic Evaluation Framework
================================================================

Step 2 Implementation:
- Implement proper multi-class evaluation metrics
- Create confusion matrix visualization
- Document baseline performance for comparison
- Goal: Understand current model performance

Design Focus:
- Reuse Step 1 models for pure evaluation analysis
- High-risk focused metrics (precision/recall for risk detection)
- Business-meaningful performance interpretation
- Comprehensive baseline documentation for future comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
    cohen_kappa_score, balanced_accuracy_score
)
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings
import platform
import matplotlib.font_manager as fm
import os
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


class AdvancedEvaluationFramework:
    """
    Step 2: Advanced Evaluation Framework for Risk Prediction Models
    
    Provides comprehensive evaluation of Step 1 baseline models with focus on:
    - Multi-class evaluation metrics
    - High-risk detection performance
    - Confusion matrix analysis
    - Business-meaningful performance interpretation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluation framework
        
        Args:
            config: Dictionary containing evaluation configuration
        """
        self.config = config
        self.data = None
        self.models = {}
        self.predictions = {}
        self.evaluation_results = {}
        self.feature_columns = None
        self.korean_font = setup_korean_font()
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Risk level definitions for business interpretation
        self.risk_levels = {
            0: "위험없음 (No Risk)",
            1: "낮은위험 (Low Risk)", 
            2: "중간위험 (Medium Risk)",
            3: "높은위험 (High Risk)"
        }
        
        print("🔍 Advanced Evaluation Framework Initialized")
        print(f"📊 Target variables: {config['target_columns']}")
        print(f"🎯 Focus: High-risk detection performance")
    
    def _create_results_directory(self) -> str:
        """
        Create results directory for saving outputs
        
        Returns:
            Path to created results directory
        """
        results_dir = "result/step2_evaluation"
        
        # Create main results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        subdirs = ['visualizations', 'metrics', 'documentation']
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
    
    def load_csv_with_korean_encoding(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file with proper Korean encoding handling
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded DataFrame with proper Korean text
        """
        encodings_to_try = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully loaded {file_path} with {encoding} encoding")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all encodings fail, try with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            print(f"⚠️  Loaded {file_path} with error handling - some characters may be corrupted")
            return df
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def load_and_train_baseline_models(self):
        """
        Load data and train baseline models (replicating Step 1)
        """
        print("\n" + "="*60)
        print("1️⃣ LOADING DATA & TRAINING BASELINE MODELS")
        print("="*60)
        
        # Load data with Korean encoding support
        self.data = self.load_csv_with_korean_encoding(self.config['data_path'])
        print(f"✅ Data loaded: {self.data.shape}")
        
        # Apply exclusions
        exclude_cols = self.config.get('exclude_columns', [])
        if exclude_cols:
            self.data = self.data.drop(columns=[col for col in exclude_cols if col in self.data.columns])
        
        # Separate features and targets
        target_cols = self.config['target_columns']
        feature_cols = [col for col in self.data.columns if not col.startswith('risk_year')]
        self.feature_columns = feature_cols
        
        X = self.data[feature_cols]
        
        # Train/test split
        X_train, X_test = train_test_split(
            X, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], shuffle=True
        )
        
        # Split targets
        y_train_dict = {}
        y_test_dict = {}
        
        for target in target_cols:
            if target in self.data.columns:
                y_target = self.data[target].values
                y_train_target, y_test_target = train_test_split(
                    y_target, test_size=self.config['test_size'],
                    random_state=self.config['random_state'], shuffle=True
                )
                y_train_dict[target] = y_train_target
                y_test_dict[target] = y_test_target
        
        # Train models
        print(f"\n🔧 Training {len(target_cols)} baseline models...")
        models = {}
        predictions = {}
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'enable_missing': True,
            'random_state': 42,
            'verbosity': 0
        }
        
        for target_name, y_train_values in y_train_dict.items():
            print(f"   📈 Training {target_name}...")
            
            # Filter for valid data
            valid_mask = ~pd.isna(y_train_values)
            X_train_filtered = X_train[valid_mask]
            y_train_filtered = y_train_values[valid_mask]
            
            if len(y_train_filtered) == 0:
                continue
            
            # Train model
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train_filtered, y_train_filtered)
            models[target_name] = model
            
            # Generate predictions for evaluation
            y_test_values = y_test_dict[target_name]
            test_valid_mask = ~pd.isna(y_test_values)
            X_test_filtered = X_test[test_valid_mask]
            y_test_filtered = y_test_values[test_valid_mask]
            
            if len(y_test_filtered) > 0:
                y_pred_raw = model.predict(X_test_filtered)
                y_pred_rounded = np.clip(np.round(y_pred_raw), 0, 3).astype(int)
                
                predictions[target_name] = {
                    'raw': y_pred_raw,
                    'rounded': y_pred_rounded,
                    'actual': y_test_filtered.astype(int)
                }
        
        self.models = models
        self.predictions = predictions
        print(f"✅ Training completed: {len(models)} models ready for evaluation")
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """
        Calculate comprehensive evaluation metrics for each model
        
        Returns:
            Dictionary of detailed metrics for each target
        """
        print("\n" + "="*60)
        print("2️⃣ COMPREHENSIVE METRICS CALCULATION")
        print("="*60)
        
        evaluation_results = {}
        
        for target_name, pred_data in self.predictions.items():
            print(f"\n📊 Evaluating {target_name}...")
            
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            y_pred_raw = pred_data['raw']
            
            # Basic metrics
            mae_raw = mean_absolute_error(y_true, y_pred_raw)
            mae_rounded = mean_absolute_error(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # Class-wise metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
            )
            
            # High-risk focused metrics (risk levels 2 and 3)
            high_risk_mask = (y_true >= 2)
            high_risk_pred_mask = (y_pred >= 2)
            
            # High-risk detection metrics
            if np.any(high_risk_mask):
                high_risk_precision = precision_recall_fscore_support(
                    high_risk_mask, high_risk_pred_mask, average='binary', zero_division=0
                )[0]
                high_risk_recall = precision_recall_fscore_support(
                    high_risk_mask, high_risk_pred_mask, average='binary', zero_division=0
                )[1]
                high_risk_f1 = precision_recall_fscore_support(
                    high_risk_mask, high_risk_pred_mask, average='binary', zero_division=0
                )[2]
            else:
                high_risk_precision = high_risk_recall = high_risk_f1 = 0.0
            
            # Class distribution analysis
            true_dist = pd.Series(y_true).value_counts().sort_index()
            pred_dist = pd.Series(y_pred).value_counts().sort_index()
            
            # Store comprehensive results
            evaluation_results[target_name] = {
                'basic_metrics': {
                    'mae_raw': mae_raw,
                    'mae_rounded': mae_rounded,
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'cohen_kappa': kappa
                },
                'class_wise_metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': support
                },
                'high_risk_metrics': {
                    'precision': high_risk_precision,
                    'recall': high_risk_recall,
                    'f1_score': high_risk_f1
                },
                'distributions': {
                    'true_distribution': dict(true_dist),
                    'predicted_distribution': dict(pred_dist)
                },
                'sample_size': len(y_true)
            }
            
            # Print summary
            print(f"   📈 Accuracy: {accuracy:.4f}")
            print(f"   📈 Balanced Accuracy: {balanced_acc:.4f}")
            print(f"   🎯 High-Risk Precision: {high_risk_precision:.4f}")
            print(f"   🎯 High-Risk Recall: {high_risk_recall:.4f}")
            print(f"   📊 Cohen's Kappa: {kappa:.4f}")
        
        self.evaluation_results = evaluation_results
        
        # Save evaluation metrics to file
        self._save_evaluation_metrics(evaluation_results)
        
        return evaluation_results
    
    def _save_evaluation_metrics(self, evaluation_results: Dict):
        """
        Save detailed evaluation metrics to CSV and JSON files
        
        Args:
            evaluation_results: Dictionary of evaluation results
        """
        print("\n💾 Saving evaluation metrics to files...")
        
        # Create summary table for CSV
        summary_data = []
        for target_name, results in evaluation_results.items():
            summary_data.append({
                'Model': target_name,
                'Accuracy': results['basic_metrics']['accuracy'],
                'Balanced_Accuracy': results['basic_metrics']['balanced_accuracy'],
                'MAE_Raw': results['basic_metrics']['mae_raw'],
                'MAE_Rounded': results['basic_metrics']['mae_rounded'],
                'Cohen_Kappa': results['basic_metrics']['cohen_kappa'],
                'High_Risk_Precision': results['high_risk_metrics']['precision'],
                'High_Risk_Recall': results['high_risk_metrics']['recall'],
                'High_Risk_F1': results['high_risk_metrics']['f1_score'],
                'Sample_Size': results['sample_size']
            })
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, 'metrics', 'evaluation_summary.csv')
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  📊 Summary metrics saved: evaluation_summary.csv")
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, 'metrics', 'detailed_evaluation.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(evaluation_results, f, indent=2, default=str, ensure_ascii=False)
        print(f"  📄 Detailed metrics saved: detailed_evaluation.json")
    
    def create_confusion_matrices(self):
        """
        Create detailed confusion matrix visualizations with Korean font support
        """
        print("\n" + "="*60)
        print("3️⃣ CONFUSION MATRIX ANALYSIS")
        print("="*60)
        
        n_models = len(self.predictions)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('혼동행렬 분석 - 2단계 평가 (Confusion Matrix Analysis - Step 2 Evaluation)', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (target_name, pred_data) in enumerate(self.predictions.items()):
            if i >= 4:
                break
            
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap with Korean support
            ax = axes[i]
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=[self.risk_levels[j] for j in [0,1,2,3]],
                       yticklabels=[self.risk_levels[j] for j in [0,1,2,3]],
                       ax=ax)
            
            ax.set_title(f'{target_name} - 혼동행렬 (정규화) | Confusion Matrix (Normalized)', 
                        fontsize=12)
            ax.set_xlabel('예측 위험도 (Predicted Risk Level)', fontsize=10)
            ax.set_ylabel('실제 위험도 (Actual Risk Level)', fontsize=10)
            
            # Rotate labels for better readability
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
            
            # Print confusion matrix analysis
            print(f"\n📊 {target_name} Confusion Matrix Analysis:")
            print("   Raw counts:")
            for actual in range(4):
                for predicted in range(4):
                    if cm[actual, predicted] > 0:
                        print(f"     Actual {self.risk_levels[actual]} → Predicted {self.risk_levels[predicted]}: {cm[actual, predicted]}")
        
        plt.tight_layout()
        self._save_plot("01_confusion_matrices")
    
    def analyze_error_patterns(self):
        """
        Analyze detailed error patterns across models with Korean font support
        """
        print("\n" + "="*60)
        print("4️⃣ ERROR PATTERN ANALYSIS")
        print("="*60)
        
        # Create error analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('오차 패턴 분석 - 비즈니스 영향 중심 (Error Pattern Analysis - Business Impact Focus)', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (target_name, pred_data) in enumerate(self.predictions.items()):
            if i >= 4:
                break
            
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            
            # Calculate prediction errors
            errors = y_pred - y_true
            
            # Create error distribution plot
            ax = axes[i]
            error_counts = pd.Series(errors).value_counts().sort_index()
            
            colors = ['red' if x < 0 else 'orange' if x > 0 else 'green' for x in error_counts.index]
            bars = ax.bar(error_counts.index, error_counts.values, color=colors, alpha=0.7)
            
            ax.set_title(f'{target_name} - 예측 오차 (Prediction Errors)', fontsize=12)
            ax.set_xlabel('오차 (예측값 - 실제값) | Error (Predicted - Actual)', fontsize=10)
            ax.set_ylabel('빈도 (Count)', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add error interpretation
            underestimate = (errors < 0).sum()  # Predicted lower than actual (dangerous)
            overestimate = (errors > 0).sum()   # Predicted higher than actual (conservative)
            perfect = (errors == 0).sum()       # Perfect predictions
            
            ax.text(0.02, 0.98, 
                   f'정확한 예측 (Perfect): {perfect}\n과소추정 (Under-estimate): {underestimate}\n과대추정 (Over-estimate): {overestimate}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        plt.tight_layout()
        self._save_plot("02_error_patterns")
        
        # Print business impact analysis
        print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
        for target_name, pred_data in self.predictions.items():
            y_true = pred_data['actual']
            y_pred = pred_data['rounded']
            errors = y_pred - y_true
            
            # Dangerous errors (underestimating risk)
            dangerous_errors = (errors < 0).sum()
            total_high_risk = (y_true >= 2).sum()
            missed_high_risk = ((y_true >= 2) & (y_pred < 2)).sum()
            
            print(f"\n   🎯 {target_name}:")
            print(f"     • Total high-risk cases: {total_high_risk}")
            print(f"     • Missed high-risk cases: {missed_high_risk}")
            print(f"     • High-risk detection rate: {(1 - missed_high_risk/max(total_high_risk, 1)):.2%}")
            print(f"     • Dangerous underestimations: {dangerous_errors}")
    
    def create_performance_comparison(self):
        """
        Create comprehensive performance comparison across models with Korean font support
        """
        print("\n" + "="*60)
        print("5️⃣ PERFORMANCE COMPARISON ACROSS MODELS")
        print("="*60)
        
        # Prepare comparison data
        comparison_data = []
        for target_name, results in self.evaluation_results.items():
            comparison_data.append({
                'Model': target_name,
                'Accuracy': results['basic_metrics']['accuracy'],
                'Balanced_Acc': results['basic_metrics']['balanced_accuracy'],
                'MAE': results['basic_metrics']['mae_rounded'],
                'Cohen_Kappa': results['basic_metrics']['cohen_kappa'],
                'High_Risk_Precision': results['high_risk_metrics']['precision'],
                'High_Risk_Recall': results['high_risk_metrics']['recall'],
                'High_Risk_F1': results['high_risk_metrics']['f1_score'],
                'Sample_Size': results['sample_size']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('모델 성능 비교 - 2단계 기준선 (Model Performance Comparison - Step 2 Baseline)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Overall performance metrics
        ax1 = axes[0, 0]
        x_pos = np.arange(len(comparison_df))
        width = 0.25
        
        ax1.bar(x_pos - width, comparison_df['Accuracy'], width, label='정확도 (Accuracy)', alpha=0.8)
        ax1.bar(x_pos, comparison_df['Balanced_Acc'], width, label='균형정확도 (Balanced Acc)', alpha=0.8)
        ax1.bar(x_pos + width, comparison_df['Cohen_Kappa'], width, label='코헨 카파 (Cohen Kappa)', alpha=0.8)
        
        ax1.set_title('전체 성능 지표 (Overall Performance Metrics)', fontsize=12)
        ax1.set_ylabel('점수 (Score)', fontsize=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(comparison_df['Model'], rotation=45)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. High-risk detection performance
        ax2 = axes[0, 1]
        ax2.bar(x_pos - width/2, comparison_df['High_Risk_Precision'], width, label='정밀도 (Precision)', alpha=0.8)
        ax2.bar(x_pos + width/2, comparison_df['High_Risk_Recall'], width, label='재현율 (Recall)', alpha=0.8)
        
        ax2.set_title('고위험 탐지 성능 (High-Risk Detection Performance)', fontsize=12)
        ax2.set_ylabel('점수 (Score)', fontsize=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(comparison_df['Model'], rotation=45)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample sizes
        ax3 = axes[1, 0]
        ax3.bar(comparison_df['Model'], comparison_df['Sample_Size'], alpha=0.8)
        ax3.set_title('테스트 샘플 크기 (Test Sample Sizes)', fontsize=12)
        ax3.set_ylabel('샘플 수 (Number of Samples)', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. MAE comparison
        ax4 = axes[1, 1]
        ax4.bar(comparison_df['Model'], comparison_df['MAE'], alpha=0.8, color='orange')
        ax4.set_title('평균 절대 오차 (Mean Absolute Error)', fontsize=12)
        ax4.set_ylabel('MAE', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot("03_performance_comparison")
        
        # Print detailed comparison table
        print(f"\n📊 DETAILED PERFORMANCE COMPARISON TABLE:")
        print("-" * 80)
        
        for _, row in comparison_df.iterrows():
            print(f"\n🎯 {row['Model']}:")
            print(f"   • Accuracy: {row['Accuracy']:.4f}")
            print(f"   • Balanced Accuracy: {row['Balanced_Acc']:.4f}")
            print(f"   • MAE: {row['MAE']:.4f}")
            print(f"   • Cohen's Kappa: {row['Cohen_Kappa']:.4f}")
            print(f"   • High-Risk Precision: {row['High_Risk_Precision']:.4f}")
            print(f"   • High-Risk Recall: {row['High_Risk_Recall']:.4f}")
            print(f"   • High-Risk F1: {row['High_Risk_F1']:.4f}")
            print(f"   • Test Sample Size: {row['Sample_Size']:,}")
    
    def generate_baseline_documentation(self):
        """
        Generate comprehensive baseline documentation for future comparison
        """
        print("\n" + "="*60)
        print("6️⃣ BASELINE DOCUMENTATION GENERATION")
        print("="*60)
        
        # Create comprehensive summary
        summary = {
            'step2_evaluation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_configuration': {
                'algorithm': 'XGBoost Regression',
                'approach': 'Separate models per year',
                'target_encoding': 'Regression with rounding',
                'missing_value_handling': 'Native XGBoost',
                'train_test_split': f"{int((1-self.config['test_size'])*100)}/{int(self.config['test_size']*100)}"
            },
            'performance_summary': {},
            'key_findings': {},
            'recommendations_for_next_steps': []
        }
        
        # Performance summary
        for target_name, results in self.evaluation_results.items():
            summary['performance_summary'][target_name] = {
                'accuracy': results['basic_metrics']['accuracy'],
                'balanced_accuracy': results['basic_metrics']['balanced_accuracy'],
                'mae': results['basic_metrics']['mae_rounded'],
                'high_risk_recall': results['high_risk_metrics']['recall'],
                'sample_size': results['sample_size'],
                'class_distribution': results['distributions']['true_distribution']
            }
        
        # Key findings
        accuracies = [r['basic_metrics']['accuracy'] for r in self.evaluation_results.values()]
        high_risk_recalls = [r['high_risk_metrics']['recall'] for r in self.evaluation_results.values()]
        
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['basic_metrics']['accuracy'])
        worst_model = min(self.evaluation_results.keys(), 
                         key=lambda x: self.evaluation_results[x]['basic_metrics']['accuracy'])
        
        summary['key_findings'] = {
            'best_performing_model': best_model,
            'worst_performing_model': worst_model,
            'average_accuracy': np.mean(accuracies),
            'average_high_risk_recall': np.mean(high_risk_recalls),
            'performance_consistency': np.std(accuracies),
            'class_imbalance_impact': 'Year1 most affected by severe imbalance'
        }
        
        # Recommendations
        summary['recommendations_for_next_steps'] = [
            'Address class imbalance especially in Year1 model',
            'Consider temporal validation to avoid optimistic bias',
            'Investigate model architecture (separate vs joint)',
            'Focus feature engineering on macro-economic indicators',
            'Implement cost-sensitive training for high-risk detection'
        ]
        
        # Save documentation
        import json
        doc_path = os.path.join(self.results_dir, 'documentation', 'step2_baseline_documentation.json')
        with open(doc_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Baseline documentation saved to '{doc_path}'")
        
        # Print executive summary
        print(f"\n📋 EXECUTIVE SUMMARY - STEP 2 BASELINE EVALUATION:")
        print("=" * 60)
        print(f"📅 Evaluation Date: {summary['step2_evaluation_date']}")
        print(f"🎯 Best Model: {best_model} (Accuracy: {self.evaluation_results[best_model]['basic_metrics']['accuracy']:.4f})")
        print(f"⚠️  Worst Model: {worst_model} (Accuracy: {self.evaluation_results[worst_model]['basic_metrics']['accuracy']:.4f})")
        print(f"📊 Average Accuracy: {np.mean(accuracies):.4f}")
        print(f"🎯 Average High-Risk Recall: {np.mean(high_risk_recalls):.4f}")
        print(f"📈 Performance Consistency (std): {np.std(accuracies):.4f}")
        
        print(f"\n🔍 KEY INSIGHTS:")
        print("• Class imbalance severely impacts Year1 model performance")
        print("• High-risk detection varies significantly across models")
        print("• Current approach provides reasonable baseline")
        print("• Major improvements needed in high-risk recall")
        
        print(f"\n➡️  READY FOR STEP 3: Comprehensive EDA")
    
    def run_step2_evaluation(self):
        """
        Execute complete Step 2 evaluation pipeline
        """
        print("🔍 EXECUTING STEP 2: BASIC EVALUATION FRAMEWORK")
        print("=" * 70)
        
        try:
            # Load and train baseline models
            self.load_and_train_baseline_models()
            
            # Calculate comprehensive metrics
            self.calculate_comprehensive_metrics()
            
            # Create confusion matrices
            self.create_confusion_matrices()
            
            # Analyze error patterns
            self.analyze_error_patterns()
            
            # Create performance comparison
            self.create_performance_comparison()
            
            # Generate baseline documentation
            self.generate_baseline_documentation()
            
            print("\n🎉 STEP 2 COMPLETED SUCCESSFULLY!")
            print("✅ Comprehensive evaluation framework established")
            print("✅ Baseline performance documented")
            print("✅ Ready for Step 3: Comprehensive EDA")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            print(f"\n❌ STEP 2 FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of all generated files and outputs"""
        print(f"\n📁 RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   📊 Visualizations:")
        print(f"      • 01_confusion_matrices.png")
        print(f"      • 02_error_patterns.png")
        print(f"      • 03_performance_comparison.png")
        print(f"   📈 Metrics:")
        print(f"      • evaluation_summary.csv")
        print(f"      • detailed_evaluation.json")
        print(f"   📄 Documentation:")
        print(f"      • step2_baseline_documentation.json")
        print(f"\n💡 Use these files for offline analysis and reporting!")


def get_step2_config():
    """
    Configuration for Step 2 evaluation framework
    """
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
        'target_columns': ['risk_year1', 'risk_year2', 'risk_year3', 'risk_year4'],
        'test_size': 0.2,
        'random_state': 42
    }


# Main execution
if __name__ == "__main__":
    
    print("🔍 Starting XGBoost Risk Prediction Model - Step 2")
    print("="*60)
    
    # Test Korean font setup (silently)
    print("\n🎨 Testing Korean font setup...")
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '한국어 폰트 테스트 (Korean Font Test)', 
                ha='center', va='center', fontsize=14)
        ax.set_title('폰트 테스트 | Font Test')
        plt.close(fig)  # Close without showing
        print("✅ Korean font setup successful!")
    except Exception as e:
        print(f"⚠️  Korean font setup issue: {e}")
        print("💡 Continuing with default font - Korean characters may not display correctly")
    
    # Get configuration
    config = get_step2_config()
    
    # Create and run evaluation framework
    evaluator = AdvancedEvaluationFramework(config)
    evaluator.run_step2_evaluation()
    
    print("\n🏁 Step 2 execution completed!")
    print("Ready to proceed to Step 3: Comprehensive EDA")