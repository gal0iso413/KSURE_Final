"""
Step 6: Model Architecture Experiments
=====================================

This script implements comprehensive model architecture experiments to find the optimal
approach before feature engineering and hyperparameter tuning.

Experiments:
1. Unified vs Individual Models
2. Classification vs Regression approaches  
3. Basic Ensemble Methods
4. Temporal Model Variants

Based on Step 5 results, using SMOTEENN as default imbalance strategy.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Imbalance handling
from imblearn.combine import SMOTEENN

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class ModelArchitectureExperiments:
    def __init__(self, data_path="dataset/credit_risk_dataset.csv"):
        """Initialize the architecture experiments."""
        self.data_path = data_path
        self.results = {
            'metadata': {
                'execution_date': datetime.now().isoformat(),
                'best_architecture': None,
                'imbalance_strategy': 'SMOTEENN'
            },
            'architecture_comparison': {},
            'detailed_results': {},
            'computational_efficiency': {}
        }
        
        # Create results directory
        self.results_dir = "result/step6_architecture"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        
        print("🚀 Step 6: Model Architecture Experiments")
        print("=" * 50)
        
    def load_and_prepare_data(self):
        """Load data and prepare for experiments using Step 4 temporal split."""
        print("\n📊 Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Define columns to exclude (from Step 5 configuration)
        exclude_columns = [
            "사업자등록번호", "대상자명", "대상자등록이력일시", "대상자기본주소",
            "청약번호", "청약상태코드", "수출자대상자번호", "특별출연협약코드", "업종코드1"
        ]
        
        # Define target columns
        self.target_columns = ["risk_year1", "risk_year2", "risk_year3", "risk_year4"]
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in exclude_columns + self.target_columns]
        
        # Handle missing values (simple approach for architecture testing)
        df_processed = df[feature_columns + self.target_columns].copy()
        
        # Fill missing values
        for col in feature_columns:
            if df_processed[col].dtype in ['object', 'category']:
                df_processed[col] = df_processed[col].fillna('Unknown')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Encode categorical variables
        label_encoders = {}
        for col in feature_columns:
            if df_processed[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Temporal split (using Step 4 approach)
        if '보험청약일자' in df.columns:
            df['보험청약일자'] = pd.to_datetime(df['보험청약일자'])
            df_sorted = df.sort_values('보험청약일자')
            split_idx = int(len(df_sorted) * 0.8)
            
            train_indices = df_sorted.index[:split_idx]
            test_indices = df_sorted.index[split_idx:]
            
            self.X_train = df_processed.loc[train_indices, feature_columns]
            self.X_test = df_processed.loc[test_indices, feature_columns]
            self.y_train = df_processed.loc[train_indices, self.target_columns]
            self.y_test = df_processed.loc[test_indices, self.target_columns]
        else:
            # Fallback to random split
            X = df_processed[feature_columns]
            y = df_processed[self.target_columns]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y['risk_year1']
            )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Target distribution in training:")
        for target in self.target_columns:
            print(f"  {target}: {self.y_train[target].value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def apply_smoteenn(self, X, y_single):
        """Apply SMOTEENN to single target variable."""
        try:
            smoteenn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y_single)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"SMOTEENN failed: {e}, using original data")
            return X, y_single
    
    def evaluate_model(self, y_true, y_pred, model_name, target_name):
        """Evaluate model performance with NaN handling."""
        # Handle NaN values in y_true
        if hasattr(y_true, 'isna'):
            valid_mask = ~y_true.isna()
        else:
            valid_mask = ~pd.isna(y_true)
        
        if valid_mask.sum() == 0:
            return {
                'error': 'No valid data for evaluation',
                'available_samples': 0,
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'recall_macro': 0.0,
                'mae': 0.0,
                'high_risk_recall': 0.0
            }
        
        # Filter to valid data only
        y_true_clean = y_true[valid_mask] if hasattr(y_true, 'iloc') else y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        # Ensure predictions are integers
        y_pred_clean = np.round(y_pred_clean).astype(int)
        y_pred_clean = np.clip(y_pred_clean, 0, y_true_clean.max())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        f1_macro = f1_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_clean, y_pred_clean, average='macro', zero_division=0)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # High-risk recall (classes 2, 3+)
        high_risk_mask = y_true_clean >= 2
        if high_risk_mask.sum() > 0:
            high_risk_recall = recall_score(
                high_risk_mask, y_pred_clean >= 2, zero_division=0
            )
        else:
            high_risk_recall = 0.0
        
        return {
            'available_samples': valid_mask.sum(),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'recall_macro': recall_macro,
            'mae': mae,
            'high_risk_recall': high_risk_recall
        }
    
    def analyze_data_availability(self):
        """Analyze data availability for each target."""
        print("\n📊 Data Availability Analysis")
        print("-" * 40)
        
        availability_report = {}
        
        for target in self.target_columns:
            train_available = (~self.y_train[target].isna()).sum()
            test_available = (~self.y_test[target].isna()).sum()
            train_pct = train_available / len(self.y_train) * 100
            test_pct = test_available / len(self.y_test) * 100
            
            availability_report[target] = {
                'train_available': train_available,
                'test_available': test_available,
                'train_percentage': train_pct,
                'test_percentage': test_pct
            }
            
            print(f"  {target}: Train={train_available}/{len(self.y_train)} ({train_pct:.1f}%), Test={test_available}/{len(self.y_test)} ({test_pct:.1f}%)")
        
        # Intersection analysis for unified models
        all_available_train = self.y_train.notna().all(axis=1).sum()
        all_available_test = self.y_test.notna().all(axis=1).sum()
        train_pct_unified = all_available_train / len(self.y_train) * 100
        test_pct_unified = all_available_test / len(self.y_test) * 100
        
        availability_report['unified_intersection'] = {
            'train_available': all_available_train,
            'test_available': all_available_test,
            'train_percentage': train_pct_unified,
            'test_percentage': test_pct_unified
        }
        
        print(f"\n  🔄 Unified (Intersection): Train={all_available_train}/{len(self.y_train)} ({train_pct_unified:.1f}%), Test={all_available_test}/{len(self.y_test)} ({test_pct_unified:.1f}%)")
        print(f"  ⚠️  Data loss for unified model: {len(self.y_train) - all_available_train} train samples, {len(self.y_test) - all_available_test} test samples")
        
        return availability_report
    
    def experiment_unified_vs_individual(self):
        """Experiment 1: Individual vs Unified Models (Honest Comparison)."""
        print("\n🔬 Experiment 1: Individual vs Unified Models")
        print("Individual: Each target uses ALL its available data")
        print("Unified: Uses intersection (ALL targets available)")
        print("-" * 50)
        
        # Analyze data availability first
        availability = self.analyze_data_availability()
        
        results = {
            'data_availability': availability
        }
        
        # Individual Models Approach (Target-specific filtering)
        print("\n🎯 Testing Individual Models (Target-specific data)...")
        individual_results = {}
        individual_models = {}
        
        for target in self.target_columns:
            print(f"  Training model for {target}...")
            
            # Use only samples where THIS target is available
            target_available_mask = ~self.y_train[target].isna()
            X_target = self.X_train[target_available_mask]
            y_target = self.y_train[target][target_available_mask]
            
            print(f"    Using {len(X_target)}/{len(self.X_train)} training samples")
            
            # Apply SMOTEENN on clean data
            X_resampled, y_resampled = self.apply_smoteenn(X_target, y_target)
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                random_state=42,
                verbosity=0,
                n_estimators=100,
                enable_missing=True
            )
            
            model.fit(X_resampled, y_resampled)
            individual_models[target] = model
            
            # Predict and evaluate only on available test data
            test_mask = ~self.y_test[target].isna()
            if test_mask.sum() > 0:
                y_pred = model.predict(self.X_test[test_mask])
                metrics = self.evaluate_model(
                    self.y_test[target][test_mask], y_pred, 'individual', target
                )
            else:
                metrics = {'error': 'No test data available', 'available_samples': 0}
            
            individual_results[target] = metrics
            print(f"    F1: {metrics.get('f1_macro', 0):.3f}, HR Recall: {metrics.get('high_risk_recall', 0):.3f}, Samples: {metrics.get('available_samples', 0)}")
        
        # Calculate average performance for individual models (only valid metrics)
        valid_individual_results = {k: v for k, v in individual_results.items() if 'error' not in v}
        if valid_individual_results:
            avg_individual = {}
            for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                values = [result[metric] for result in valid_individual_results.values() if metric in result]
                avg_individual[metric] = np.mean(values) if values else 0.0
        else:
            avg_individual = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
        
        results['individual_models'] = {
            'avg_performance': avg_individual,
            'detailed_results': individual_results
        }
        
        # Unified Model Approach (True Intersection)
        print("\n🔄 Testing Unified Model (Intersection approach)...")
        
        # Only use samples where ALL targets are available
        all_targets_available = self.y_train.notna().all(axis=1)
        X_unified = self.X_train[all_targets_available]
        y_unified = self.y_train[all_targets_available]
        
        print(f"  Using {len(X_unified)}/{len(self.X_train)} training samples (intersection)")
        print(f"  Data loss: {len(self.X_train) - len(X_unified)} samples ({(len(self.X_train) - len(X_unified))/len(self.X_train)*100:.1f}%)")
        
        if len(X_unified) > 0:
            # Create multi-output classifier
            base_model = xgb.XGBClassifier(
                random_state=42,
                verbosity=0,
                n_estimators=100,
                enable_missing=True
            )
            
            unified_model = MultiOutputClassifier(base_model)
            unified_model.fit(X_unified, y_unified)
            
            # Predict and evaluate (fair comparison on same test data as individual models)
            unified_results = {}
            for i, target in enumerate(self.target_columns):
                test_mask = ~self.y_test[target].isna()
                if test_mask.sum() > 0:
                    y_pred_unified = unified_model.predict(self.X_test[test_mask])
                    metrics = self.evaluate_model(
                        self.y_test[target][test_mask], y_pred_unified[:, i], 'unified', target
                    )
                else:
                    metrics = {'error': 'No test data available', 'available_samples': 0}
                
                unified_results[target] = metrics
                print(f"    {target} - F1: {metrics.get('f1_macro', 0):.3f}, HR Recall: {metrics.get('high_risk_recall', 0):.3f}, Samples: {metrics.get('available_samples', 0)}")
            
            # Calculate average performance for unified model
            valid_unified_results = {k: v for k, v in unified_results.items() if 'error' not in v}
            if valid_unified_results:
                avg_unified = {}
                for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                    values = [result[metric] for result in valid_unified_results.values() if metric in result]
                    avg_unified[metric] = np.mean(values) if values else 0.0
            else:
                avg_unified = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
        else:
            print("  ❌ No samples available for unified training!")
            unified_results = {target: {'error': 'No training data available'} for target in self.target_columns}
            avg_unified = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
            unified_model = None
        
        results['unified_model'] = {
            'avg_performance': avg_unified,
            'detailed_results': unified_results
        }
        
        # Compare and determine winner
        individual_score = avg_individual['f1_macro'] * 0.6 + avg_individual['high_risk_recall'] * 0.4
        unified_score = avg_unified['f1_macro'] * 0.6 + avg_unified['high_risk_recall'] * 0.4
        
        winner = 'individual_models' if individual_score > unified_score else 'unified_model'
        results['winner'] = winner
        results['comparison_scores'] = {
            'individual_score': individual_score,
            'unified_score': unified_score
        }
        
        print(f"\n🏆 Results Comparison:")
        print(f"Individual Models - F1: {avg_individual['f1_macro']:.3f}, HR Recall: {avg_individual['high_risk_recall']:.3f}, Combined Score: {individual_score:.3f}")
        print(f"Unified Model - F1: {avg_unified['f1_macro']:.3f}, HR Recall: {avg_unified['high_risk_recall']:.3f}, Combined Score: {unified_score:.3f}")
        print(f"Winner: {winner}")
        
        if winner == 'individual_models':
            print("\n💡 Individual models won - this is expected due to optimal data usage per target")
        else:
            print("\n💡 Unified model won - despite data loss, it captured inter-target relationships")
        
        self.results['detailed_results']['unified_vs_individual'] = results
        return results, individual_models if winner == 'individual_models' else unified_model
    
    def experiment_classification_vs_regression(self, best_architecture, best_models):
        """Experiment 2: Classification vs Regression."""
        print("\n🔬 Experiment 2: Classification vs Regression")
        print("-" * 40)
        
        results = {}
        
        # We already have classification results from Experiment 1
        if isinstance(best_models, dict):  # Individual models
            classification_results = self.results['detailed_results']['unified_vs_individual']['individual_models']
        else:  # Unified model
            classification_results = self.results['detailed_results']['unified_vs_individual']['unified_model']
        
        results['classification'] = classification_results
        
        # Test Regression Approach
        print("Testing Regression Approach...")
        
        if best_architecture == 'individual_models':
            regression_results = {}
            regression_models = {}
            
            for target in self.target_columns:
                print(f"  Training regression model for {target}...")
                
                # Use target-specific filtering (same as classification)
                target_available_mask = ~self.y_train[target].isna()
                X_target = self.X_train[target_available_mask]
                y_target = self.y_train[target][target_available_mask]
                
                # Apply SMOTEENN (for consistency)
                X_resampled, y_resampled = self.apply_smoteenn(X_target, y_target)
                
                # Train XGBoost Regressor
                model = xgb.XGBRegressor(
                    random_state=42,
                    verbosity=0,
                    n_estimators=100,
                    enable_missing=True
                )
                
                model.fit(X_resampled, y_resampled)
                regression_models[target] = model
                
                # Predict and evaluate on available test data
                test_mask = ~self.y_test[target].isna()
                if test_mask.sum() > 0:
                    y_pred = model.predict(self.X_test[test_mask])
                    metrics = self.evaluate_model(
                        self.y_test[target][test_mask], y_pred, 'regression', target
                    )
                else:
                    metrics = {'error': 'No test data available', 'available_samples': 0}
                
                regression_results[target] = metrics
            
            # Calculate average performance (only valid metrics)
            valid_regression_results = {k: v for k, v in regression_results.items() if 'error' not in v}
            if valid_regression_results:
                avg_regression = {}
                for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                    values = [result[metric] for result in valid_regression_results.values() if metric in result]
                    avg_regression[metric] = np.mean(values) if values else 0.0
            else:
                avg_regression = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
            
            best_regression_models = regression_models
            
        else:  # Unified model
            # Use intersection approach for unified regressor
            all_targets_available = self.y_train.notna().all(axis=1)
            X_unified = self.X_train[all_targets_available]
            y_unified = self.y_train[all_targets_available]
            
            if len(X_unified) > 0:
                # Create multi-output regressor
                base_model = xgb.XGBRegressor(
                    random_state=42,
                    verbosity=0,
                    n_estimators=100,
                    enable_missing=True
                )
                
                unified_regressor = MultiOutputRegressor(base_model)
                unified_regressor.fit(X_unified, y_unified)
                
                # Predict and evaluate
                regression_results = {}
                for i, target in enumerate(self.target_columns):
                    test_mask = ~self.y_test[target].isna()
                    if test_mask.sum() > 0:
                        y_pred_unified = unified_regressor.predict(self.X_test[test_mask])
                        metrics = self.evaluate_model(
                            self.y_test[target][test_mask], y_pred_unified[:, i], 'regression', target
                        )
                    else:
                        metrics = {'error': 'No test data available', 'available_samples': 0}
                    regression_results[target] = metrics
                
                # Calculate average performance
                valid_regression_results = {k: v for k, v in regression_results.items() if 'error' not in v}
                if valid_regression_results:
                    avg_regression = {}
                    for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                        values = [result[metric] for result in valid_regression_results.values() if metric in result]
                        avg_regression[metric] = np.mean(values) if values else 0.0
                else:
                    avg_regression = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
                
                best_regression_models = unified_regressor
            else:
                print("  ❌ No samples available for unified regression training!")
                regression_results = {target: {'error': 'No training data available'} for target in self.target_columns}
                avg_regression = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
                best_regression_models = None
        
        results['regression'] = {
            'avg_performance': avg_regression,
            'detailed_results': regression_results
        }
        
        # Compare classification vs regression
        classification_score = classification_results['avg_performance']['f1_macro'] * 0.6 + \
                              classification_results['avg_performance']['high_risk_recall'] * 0.4
        regression_score = avg_regression['f1_macro'] * 0.6 + avg_regression['high_risk_recall'] * 0.4
        
        winner = 'classification' if classification_score > regression_score else 'regression'
        results['winner'] = winner
        results['comparison_scores'] = {
            'classification_score': classification_score,
            'regression_score': regression_score
        }
        
        print(f"\n🏆 Classification vs Regression Results:")
        print(f"Classification - F1: {classification_results['avg_performance']['f1_macro']:.3f}, HR Recall: {classification_results['avg_performance']['high_risk_recall']:.3f}, Score: {classification_score:.3f}")
        print(f"Regression - F1: {avg_regression['f1_macro']:.3f}, HR Recall: {avg_regression['high_risk_recall']:.3f}, Score: {regression_score:.3f}")
        print(f"Winner: {winner}")
        
        self.results['detailed_results']['classification_vs_regression'] = results
        return results, best_regression_models if winner == 'regression' else best_models
    
    def experiment_ensemble_methods(self, best_architecture, best_approach, best_models):
        """Experiment 3: Basic Ensemble Methods."""
        print("\n🔬 Experiment 3: Basic Ensemble Methods")
        print("-" * 40)
        
        results = {}
        
        # Only test ensemble if individual models performed well
        if best_architecture != 'individual_models':
            print("Skipping ensemble - only applicable for individual models")
            results['skipped'] = True
            results['reason'] = 'Only applicable for individual models'
            self.results['detailed_results']['ensemble_methods'] = results
            return results, best_models
        
        print("Testing Simple Voting Ensemble...")
        
        ensemble_results = {}
        
        for target in self.target_columns:
            print(f"  Creating ensemble for {target}...")
            
            # Use target-specific filtering (same as individual models)
            target_available_mask = ~self.y_train[target].isna()
            X_target = self.X_train[target_available_mask]
            y_target = self.y_train[target][target_available_mask]
            
            try:
                # Create simple ensemble by training multiple models and averaging predictions
                models = []
                
                if best_approach == 'classification':
                    model1 = xgb.XGBClassifier(random_state=42, verbosity=0, n_estimators=50)
                    model2 = xgb.XGBClassifier(random_state=123, verbosity=0, n_estimators=100)
                    model3 = xgb.XGBClassifier(random_state=456, verbosity=0, n_estimators=150)
                    models = [model1, model2, model3]
                else:
                    model1 = xgb.XGBRegressor(random_state=42, verbosity=0, n_estimators=50)
                    model2 = xgb.XGBRegressor(random_state=123, verbosity=0, n_estimators=100)
                    model3 = xgb.XGBRegressor(random_state=456, verbosity=0, n_estimators=150)
                    models = [model1, model2, model3]
                
                # Apply SMOTEENN on target-specific data
                X_resampled, y_resampled = self.apply_smoteenn(X_target, y_target)
                
                # Train all models
                for model in models:
                    model.fit(X_resampled, y_resampled)
                
                # Predict and evaluate on available test data
                test_mask = ~self.y_test[target].isna()
                if test_mask.sum() > 0:
                    # Get predictions from all models
                    predictions = []
                    for model in models:
                        if best_approach == 'classification':
                            pred = model.predict_proba(self.X_test[test_mask])
                            predictions.append(pred)
                        else:
                            pred = model.predict(self.X_test[test_mask])
                            predictions.append(pred)
                    
                    # Average predictions
                    if best_approach == 'classification':
                        # Average probabilities and take argmax
                        avg_proba = np.mean(predictions, axis=0)
                        y_pred = np.argmax(avg_proba, axis=1)
                    else:
                        # Average regression predictions
                        y_pred = np.mean(predictions, axis=0)
                    
                    metrics = self.evaluate_model(
                        self.y_test[target][test_mask], y_pred, 'ensemble', target
                    )
                else:
                    metrics = {'error': 'No test data available', 'available_samples': 0}
                
            except Exception as e:
                print(f"    ❌ Ensemble failed for {target}: {str(e)}")
                metrics = {'error': f'Ensemble failed: {str(e)}', 'available_samples': 0}
            
            ensemble_results[target] = metrics
        
        # Calculate average performance (only valid metrics)
        valid_ensemble_results = {k: v for k, v in ensemble_results.items() if 'error' not in v}
        if valid_ensemble_results:
            avg_ensemble = {}
            for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                values = [result[metric] for result in valid_ensemble_results.values() if metric in result]
                avg_ensemble[metric] = np.mean(values) if values else 0.0
        else:
            avg_ensemble = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
        
        results['ensemble'] = {
            'avg_performance': avg_ensemble,
            'detailed_results': ensemble_results
        }
        
        # Compare with best single model approach
        single_model_results = self.results['detailed_results']['classification_vs_regression'][best_approach]
        single_model_score = single_model_results['avg_performance']['f1_macro'] * 0.6 + \
                            single_model_results['avg_performance']['high_risk_recall'] * 0.4
        ensemble_score = avg_ensemble['f1_macro'] * 0.6 + avg_ensemble['high_risk_recall'] * 0.4
        
        winner = 'ensemble' if ensemble_score > single_model_score else 'single_model'
        results['winner'] = winner
        results['comparison_scores'] = {
            'single_model_score': single_model_score,
            'ensemble_score': ensemble_score
        }
        
        print(f"Winner: {winner}")
        print(f"Single Model - F1: {single_model_results['avg_performance']['f1_macro']:.3f}, HR Recall: {single_model_results['avg_performance']['high_risk_recall']:.3f}")
        print(f"Ensemble - F1: {avg_ensemble['f1_macro']:.3f}, HR Recall: {avg_ensemble['high_risk_recall']:.3f}")
        
        self.results['detailed_results']['ensemble_methods'] = results
        return results, best_models  # For simplicity, return original models
    
    def experiment_temporal_variants(self, best_models):
        """Experiment 4: Temporal Model Variants."""
        print("\n🔬 Experiment 4: Temporal Model Variants")
        print("-" * 40)
        
        results = {}
        
        # Add temporal features
        print("Adding temporal features...")
        
        # Create temporal features (simplified for this step)
        X_train_temporal = self.X_train.copy()
        X_test_temporal = self.X_test.copy()
        
        # Add simple temporal features (year index for each target)
        for i, target in enumerate(self.target_columns):
            X_train_temporal[f'target_year_{target}'] = i + 1
            X_test_temporal[f'target_year_{target}'] = i + 1
        
        # Add rolling features (simplified)
        numeric_columns = [col for col in self.X_train.columns[:5] if self.X_train[col].dtype in ['int64', 'float64']]
        for col in numeric_columns:
            X_train_temporal[f'{col}_rolling_mean'] = self.X_train[col].rolling(window=3, min_periods=1).mean()
            X_test_temporal[f'{col}_rolling_mean'] = self.X_test[col].rolling(window=3, min_periods=1).mean()
        
        print(f"Added temporal features. New shape: {X_train_temporal.shape}")
        
        # Test temporal models (using best architecture from previous experiments)
        temporal_results = {}
        
        if isinstance(best_models, dict):  # Individual models
            for target in self.target_columns:
                print(f"  Training temporal model for {target}...")
                
                # Use target-specific filtering
                target_available_mask = ~self.y_train[target].isna()
                X_target_temporal = X_train_temporal[target_available_mask]
                y_target = self.y_train[target][target_available_mask]
                
                # Apply SMOTEENN
                X_resampled, y_resampled = self.apply_smoteenn(X_target_temporal, y_target)
                
                # Train XGBoost with temporal features
                model = xgb.XGBClassifier(
                    random_state=42,
                    verbosity=0,
                    n_estimators=100,
                    enable_missing=True
                )
                
                model.fit(X_resampled, y_resampled)
                
                # Predict and evaluate on available test data
                test_mask = ~self.y_test[target].isna()
                if test_mask.sum() > 0:
                    y_pred = model.predict(X_test_temporal[test_mask])
                    metrics = self.evaluate_model(
                        self.y_test[target][test_mask], y_pred, 'temporal', target
                    )
                else:
                    metrics = {'error': 'No test data available', 'available_samples': 0}
                
                temporal_results[target] = metrics
                
        else:  # Unified model
            # Use intersection approach for unified temporal model
            all_targets_available = self.y_train.notna().all(axis=1)
            X_unified_temporal = X_train_temporal[all_targets_available]
            y_unified = self.y_train[all_targets_available]
            
            if len(X_unified_temporal) > 0:
                unified_temporal = MultiOutputClassifier(
                    xgb.XGBClassifier(
                        random_state=42,
                        verbosity=0,
                        n_estimators=100,
                        enable_missing=True
                    )
                )
                
                unified_temporal.fit(X_unified_temporal, y_unified)
                
                for i, target in enumerate(self.target_columns):
                    test_mask = ~self.y_test[target].isna()
                    if test_mask.sum() > 0:
                        y_pred_unified = unified_temporal.predict(X_test_temporal[test_mask])
                        metrics = self.evaluate_model(
                            self.y_test[target][test_mask], y_pred_unified[:, i], 'temporal', target
                        )
                    else:
                        metrics = {'error': 'No test data available', 'available_samples': 0}
                    temporal_results[target] = metrics
            else:
                print("  ❌ No samples available for unified temporal training!")
                temporal_results = {target: {'error': 'No training data available'} for target in self.target_columns}
        
        # Calculate average performance (only valid metrics)
        valid_temporal_results = {k: v for k, v in temporal_results.items() if 'error' not in v}
        if valid_temporal_results:
            avg_temporal = {}
            for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']:
                values = [result[metric] for result in valid_temporal_results.values() if metric in result]
                avg_temporal[metric] = np.mean(values) if values else 0.0
        else:
            avg_temporal = {metric: 0.0 for metric in ['accuracy', 'f1_macro', 'recall_macro', 'mae', 'high_risk_recall']}
        
        results['temporal_enhanced'] = {
            'avg_performance': avg_temporal,
            'detailed_results': temporal_results
        }
        
        print(f"Temporal Enhanced - F1: {avg_temporal['f1_macro']:.3f}, HR Recall: {avg_temporal['high_risk_recall']:.3f}")
        
        self.results['detailed_results']['temporal_variants'] = results
        return results
    
    def create_comparison_summary(self):
        """Create comprehensive comparison summary."""
        print("\n📊 Creating Comparison Summary...")
        
        comparison_data = []
        
        # Extract results from all experiments
        experiments = self.results['detailed_results']
        
        # Unified vs Individual
        if 'unified_vs_individual' in experiments:
            exp = experiments['unified_vs_individual']
            for approach in ['individual_models', 'unified_model']:
                if approach in exp:
                    perf = exp[approach]['avg_performance']
                    comparison_data.append({
                        'architecture': approach,
                        'approach': 'classification',
                        'f1_macro': perf['f1_macro'],
                        'high_risk_recall': perf['high_risk_recall'],
                        'recall_macro': perf['recall_macro'],
                        'combined_score': perf['f1_macro'] * 0.6 + perf['high_risk_recall'] * 0.4
                    })
        
        # Classification vs Regression
        if 'classification_vs_regression' in experiments:
            exp = experiments['classification_vs_regression']
            for approach in ['classification', 'regression']:
                if approach in exp and 'avg_performance' in exp[approach]:
                    perf = exp[approach]['avg_performance']
                    comparison_data.append({
                        'architecture': 'best_from_exp1',
                        'approach': approach,
                        'f1_macro': perf['f1_macro'],
                        'high_risk_recall': perf['high_risk_recall'],
                        'recall_macro': perf['recall_macro'],
                        'combined_score': perf['f1_macro'] * 0.6 + perf['high_risk_recall'] * 0.4
                    })
        
        # Ensemble
        if 'ensemble_methods' in experiments and 'ensemble' in experiments['ensemble_methods']:
            perf = experiments['ensemble_methods']['ensemble']['avg_performance']
            comparison_data.append({
                'architecture': 'ensemble',
                'approach': 'ensemble',
                'f1_macro': perf['f1_macro'],
                'high_risk_recall': perf['high_risk_recall'],
                'recall_macro': perf['recall_macro'],
                'combined_score': perf['f1_macro'] * 0.6 + perf['high_risk_recall'] * 0.4
            })
        
        # Temporal
        if 'temporal_variants' in experiments and 'temporal_enhanced' in experiments['temporal_variants']:
            perf = experiments['temporal_variants']['temporal_enhanced']['avg_performance']
            comparison_data.append({
                'architecture': 'temporal_enhanced',
                'approach': 'temporal',
                'f1_macro': perf['f1_macro'],
                'high_risk_recall': perf['high_risk_recall'],
                'recall_macro': perf['recall_macro'],
                'combined_score': perf['f1_macro'] * 0.6 + perf['high_risk_recall'] * 0.4
            })
        
        # Sort by combined score
        comparison_data.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.results['architecture_comparison'] = comparison_data
        
        # Determine best overall architecture
        if comparison_data:
            best = comparison_data[0]
            self.results['metadata']['best_architecture'] = f"{best['architecture']}_{best['approach']}"
            
            print(f"\n🏆 Best Architecture: {best['architecture']} with {best['approach']} approach")
            print(f"   F1-Macro: {best['f1_macro']:.3f}")
            print(f"   High-Risk Recall: {best['high_risk_recall']:.3f}")
            print(f"   Combined Score: {best['combined_score']:.3f}")
        
        return comparison_data
    
    def create_visualizations(self):
        """Create comparison visualizations."""
        print("\n📈 Creating visualizations...")
        
        if not self.results['architecture_comparison']:
            print("No comparison data available for visualization")
            return
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        comparison_data = self.results['architecture_comparison']
        architectures = [f"{item['architecture']}\n{item['approach']}" for item in comparison_data]
        f1_scores = [item['f1_macro'] for item in comparison_data]
        hr_recalls = [item['high_risk_recall'] for item in comparison_data]
        combined_scores = [item['combined_score'] for item in comparison_data]
        
        # F1-Macro comparison
        ax1.bar(architectures, f1_scores, color='skyblue', alpha=0.7)
        ax1.set_title('F1-Macro Score by Architecture')
        ax1.set_ylabel('F1-Macro Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # High-Risk Recall comparison
        ax2.bar(architectures, hr_recalls, color='lightcoral', alpha=0.7)
        ax2.set_title('High-Risk Recall by Architecture')
        ax2.set_ylabel('High-Risk Recall')
        ax2.tick_params(axis='x', rotation=45)
        
        # Combined Score comparison
        ax3.bar(architectures, combined_scores, color='lightgreen', alpha=0.7)
        ax3.set_title('Combined Score by Architecture')
        ax3.set_ylabel('Combined Score (F1*0.6 + HR_Recall*0.4)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance scatter plot
        ax4.scatter(f1_scores, hr_recalls, s=100, alpha=0.7, c=combined_scores, cmap='viridis')
        for i, arch in enumerate(architectures):
            ax4.annotate(arch.replace('\n', ' '), (f1_scores[i], hr_recalls[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('F1-Macro Score')
        ax4.set_ylabel('High-Risk Recall')
        ax4.set_title('F1-Macro vs High-Risk Recall')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/plots/architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.results_dir}/plots/architecture_comparison.png")
    
    def save_results(self):
        """Save all results to JSON file."""
        print(f"\n💾 Saving results to {self.results_dir}/step6_architecture_results.json")
        
        with open(f'{self.results_dir}/step6_architecture_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'execution_date': self.results['metadata']['execution_date'],
            'best_architecture': self.results['metadata']['best_architecture'],
            'top_3_architectures': self.results['architecture_comparison'][:3] if self.results['architecture_comparison'] else [],
            'key_findings': {
                'unified_vs_individual_winner': self.results['detailed_results'].get('unified_vs_individual', {}).get('winner'),
                'classification_vs_regression_winner': self.results['detailed_results'].get('classification_vs_regression', {}).get('winner'),
                'ensemble_improvement': self.results['detailed_results'].get('ensemble_methods', {}).get('winner'),
                'temporal_enhancement': 'temporal_variants' in self.results['detailed_results']
            }
        }
        
        with open(f'{self.results_dir}/step6_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅ Results saved successfully!")
    
    def run_all_experiments(self):
        """Run all architecture experiments."""
        print("🚀 Starting Model Architecture Experiments")
        print("=" * 60)
        
        # Load data
        self.load_and_prepare_data()
        
        # Experiment 1: Unified vs Individual Models
        exp1_results, best_models = self.experiment_unified_vs_individual()
        best_architecture = exp1_results['winner']
        
        # Experiment 2: Classification vs Regression
        exp2_results, best_models = self.experiment_classification_vs_regression(
            best_architecture, best_models
        )
        best_approach = exp2_results['winner']
        
        # Experiment 3: Ensemble Methods
        exp3_results, best_models = self.experiment_ensemble_methods(
            best_architecture, best_approach, best_models
        )
        
        # Experiment 4: Temporal Variants
        exp4_results = self.experiment_temporal_variants(best_models)
        
        # Create comprehensive comparison
        self.create_comparison_summary()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n🎉 All experiments completed successfully!")
        print(f"📊 Results saved in: {self.results_dir}")
        print(f"🏆 Best Architecture: {self.results['metadata']['best_architecture']}")
        
        return self.results

def main():
    """Main execution function."""
    # Initialize experiments
    experiments = ModelArchitectureExperiments()
    
    # Run all experiments
    results = experiments.run_all_experiments()
    
    return results

if __name__ == "__main__":
    results = main()