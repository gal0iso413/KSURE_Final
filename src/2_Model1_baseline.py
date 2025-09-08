"""
XGBoost Risk Prediction Model - Step 1: Multi-Model Training
==========================================================

Step 1 Implementation (TRAINING FOCUS):
- Load data from 1_Dataset.py output
- Train 3 baseline models: XGBoost, MLP, RandomForest
- Save trained models and predictions for step1_evaluate.py
- Focus: Establish comprehensive baseline with multiple algorithm families
- Clean separation: Training only, evaluation handled by step1_evaluate.py

Design Decisions:
- 4 Separate Models per Algorithm: One for each risk_year (1,2,3,4)
- CLASSIFICATION: Direct class prediction (0,1,2,3)
- Native Missing Handling: Let XGBoost handle missing X variables
- Target-Specific Filtering: Each model uses only rows with valid data
- Standardized Output: Save models, predictions, and metadata for evaluation step
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
import os
import platform
import matplotlib.font_manager as fm
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
def setup_korean_font() -> Optional[str]:
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
        logger.info(f"Korean font set: {korean_font}")
    else:
        logger.warning("Preferred Korean fonts not found. Using fallback options...")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    return korean_font

# Set up Korean font
setup_korean_font()
plt.style.use('default')
sns.set_palette("husl")


class MultiModelTrainer:
    """
    Step 1: Multi-Model Training for Risk Prediction
    
    Trains XGBoost, MLP, and RandomForest models for baseline comparison.
    Focuses on training and saving results for subsequent evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize multi-model trainer
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = {}
        self.y_test = {}
        self.train_indices = None
        self.test_indices = None
        self.feature_columns = None
        
        # Store trained models: {algorithm: {target: model}}
        self.models = {}
        # Store predictions: {algorithm: {target: {'classes': np.array, 'actual': np.array}}}
        self.predictions = {}
        
        # Create results directory
        self.results_dir = self._create_results_directory()
        
        logger.info("Multi-Model Trainer Initialized")
        logger.info(f"Target variables: {config['target_columns']}")
        logger.info(f"Algorithms: XGBoost, MLP, RandomForest")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def _create_results_directory(self) -> str:
        """Create results directory for saving outputs"""
        results_dir = "../results/step1_baseline"
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['models', 'predictions', 'metadata']
        for subdir in subdirs:
            os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
        
        return results_dir
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
        """
        Load and preprocess data for training
        
        Returns:
            X_train, X_test, y_train_dict, y_test_dict
        """
        logger.info("DATA LOADING & PREPROCESSING")
        
        # Load data
        try:
            self.data = pd.read_csv(self.config['data_path'])
            logger.info(f"Data loaded successfully: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        # Apply column exclusions
        target_cols = self.config['target_columns']
        exclude_cols = self.config.get('exclude_columns', [])
        if exclude_cols:
            logger.info(f"Excluding {len(exclude_cols)} specified columns...")
            self.data = self.data.drop(columns=[col for col in exclude_cols if col in self.data.columns])
        
        # Separate features and targets
        feature_cols = [col for col in self.data.columns if not col.startswith('risk_year')]
        X = self.data[feature_cols]
        self.feature_columns = feature_cols
        
        logger.info(f"Dataset structure:")
        logger.info(f"   • Total records: {len(self.data):,}")
        logger.info(f"   • Feature columns: {len(feature_cols)}")
        logger.info(f"   • Target variables: {len([t for t in target_cols if t in self.data.columns])}")
        
        # Create shared train/test split (preserve indices for evaluation step)
        logger.info(f"Creating train/test split ({int((1-self.config['test_size'])*100)}/{int(self.config['test_size']*100)})...")
        indices = np.arange(len(self.data))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            shuffle=True
        )
        
        # Store indices for evaluation step
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # Create target dictionaries
        y_train_dict = {}
        y_test_dict = {}
        for target in target_cols:
            if target in self.data.columns:
                y_full = self.data[target].values
                y_train_dict[target] = y_full[train_idx]
                y_test_dict[target] = y_full[test_idx]
        
        logger.info(f"   • Training records: {len(X_train):,}")
        logger.info(f"   • Test records: {len(X_test):,}")
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_dict
        self.y_test = y_test_dict
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train_all_models(self) -> Dict:
        """
        Train all three baseline models (XGBoost, MLP, RandomForest)
        
        Returns:
            Dictionary of trained models
        """
        logger.info("MULTI-MODEL TRAINING")
        
        # Define model configurations
        algorithms = {
            'xgboost': {
                'estimator': xgb.XGBClassifier,
                'params': {
                    'objective': 'multi:softprob',
                    'num_class': 4,
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'eval_metric': 'mlogloss',
                    'enable_missing': True,
                    'random_state': self.config['random_state'],
                    'verbosity': 0,
                    'n_jobs': -1
                }
            },
            'mlp': {
                'estimator': Pipeline,
                'params': {
                    'steps': [
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler()),
                        ('mlp', MLPClassifier(
                            hidden_layer_sizes=(100, 50),
                            max_iter=500,
                            random_state=self.config['random_state'],
                            early_stopping=True,
                            validation_fraction=0.1
                        ))
                    ]
                }
            },
            'randomforest': {
                'estimator': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.config['random_state'],
                    'n_jobs': -1
                }
            }
        }
        
        # Override with config params if provided
        for alg_name in algorithms:
            config_params = self.config.get(f'{alg_name}_params', {})
            if config_params:
                algorithms[alg_name]['params'].update(config_params)
        
        logger.info(f"Training algorithms: {list(algorithms.keys())}")
        logger.info(f"Training separate models per target for each algorithm...")
        
        for alg_name, alg_config in algorithms.items():
            logger.info(f"Training algorithm: {alg_name.upper()}")
            
            per_target_models = {}
            per_target_predictions = {}
            
            for target_name, y_train_values in self.y_train.items():
                logger.info(f"Training {alg_name} for {target_name}...")
                
                try:
                    # Filter valid training data
                    valid_mask = ~pd.isna(y_train_values)
                    X_train_filtered = self.X_train[valid_mask]
                    y_train_filtered = y_train_values[valid_mask].astype(int)
                    
                    if len(y_train_filtered) == 0:
                        logger.warning(f"No valid training data for {target_name}")
                        continue
                    
                    logger.info(f"Training samples: {len(y_train_filtered):,}")
                    logger.info(f"Target classes: {sorted(np.unique(y_train_filtered))}")
                    
                    # Create and train model
                    if alg_config['estimator'] == Pipeline:
                        model = alg_config['estimator'](**alg_config['params'])
                    else:
                        model = alg_config['estimator'](**alg_config['params'])
                    
                    logger.info(f"Training {alg_name}...")
                    model.fit(X_train_filtered, y_train_filtered)
                    per_target_models[target_name] = model
                    
                    # Generate predictions on test set
                    y_test_values = self.y_test[target_name]
                    test_valid_mask = ~pd.isna(y_test_values)
                    X_test_filtered = self.X_test[test_valid_mask]
                    y_test_filtered = y_test_values[test_valid_mask].astype(int)
                    
                    if len(y_test_filtered) > 0:
                        y_pred_classes = model.predict(X_test_filtered)
                        per_target_predictions[target_name] = {
                            'classes': y_pred_classes,
                            'actual': y_test_filtered,
                            'test_indices': self.test_indices[test_valid_mask]
                        }
                        
                        # Quick training performance check
                        train_pred = model.predict(X_train_filtered)
                        train_accuracy = accuracy_score(y_train_filtered, train_pred)
                        test_accuracy = accuracy_score(y_test_filtered, y_pred_classes)
                        
                        logger.info(f"{alg_name} - Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {alg_name} for {target_name}: {e}")
                    continue
            
            self.models[alg_name] = per_target_models
            self.predictions[alg_name] = per_target_predictions
        
        total_models = sum(len(models) for models in self.models.values())
        logger.info(f"Training completed: {total_models} models across {len(self.models)} algorithms")
        
        return self.models
    
    def save_training_results(self):
        """Save all training results for evaluation step"""
        logger.info("SAVING TRAINING RESULTS")
        
        # Save trained models
        logger.info("Saving trained models...")
        for alg_name, per_target_models in self.models.items():
            for target_name, model in per_target_models.items():
                if alg_name == 'xgboost':
                    model_path = os.path.join(self.results_dir, 'models', f'{target_name}_{alg_name}_model.json')
                    model.save_model(model_path)
                else:
                    model_path = os.path.join(self.results_dir, 'models', f'{target_name}_{alg_name}_model.joblib')
                    joblib.dump(model, model_path)
                logger.info(f"Saved: {os.path.basename(model_path)}")
        
        # Save predictions in structured format
        logger.info("Saving predictions...")
        predictions_data = []
        for alg_name, per_target_preds in self.predictions.items():
            for target_name, pred_data in per_target_preds.items():
                for i, (pred_class, actual_class, test_idx) in enumerate(zip(
                    pred_data['classes'], pred_data['actual'], pred_data['test_indices']
                )):
                    predictions_data.append({
                        'algorithm': alg_name,
                        'target': target_name,
                        'test_index': int(test_idx),
                        'predicted_class': int(pred_class),
                        'actual_class': int(actual_class)
                    })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_path = os.path.join(self.results_dir, 'predictions', 'all_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved: all_predictions.csv ({len(predictions_data):,} prediction records)")
        
        # Save metadata for evaluation step
        logger.info("Saving metadata...")
        metadata = {
            'training_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_used': self.config,
            'algorithms_trained': list(self.models.keys()),
            'target_columns': self.config['target_columns'],
            'feature_columns': self.feature_columns,
            'train_indices': self.train_indices.tolist(),
            'test_indices': self.test_indices.tolist(),
            'data_shape': {
                'total_records': len(self.data),
                'n_features': len(self.feature_columns),
                'n_train': len(self.train_indices),
                'n_test': len(self.test_indices)
            },
            'model_counts': {alg: len(models) for alg, models in self.models.items()},
            'prediction_counts': {
                alg: {target: len(preds['classes']) for target, preds in per_target.items()}
                for alg, per_target in self.predictions.items()
            }
        }
        
        metadata_path = os.path.join(self.results_dir, 'metadata', 'training_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: training_metadata.json")
        
        # Create training summary
        self._create_training_summary()
    
    def _create_training_summary(self):
        """Create a quick training summary visualization"""
        print("\n📊 Creating training summary...")
        
        # Collect training statistics
        summary_data = []
        for alg_name, per_target_preds in self.predictions.items():
            for target_name, pred_data in per_target_preds.items():
                accuracy = accuracy_score(pred_data['actual'], pred_data['classes'])
                f1 = f1_score(pred_data['actual'], pred_data['classes'], average='macro')
                
                summary_data.append({
                    'Algorithm': alg_name.upper(),
                    'Target': target_name,
                    'Accuracy': accuracy,
                    'F1_Macro': f1,
                    'Test_Samples': len(pred_data['actual'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create summary visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Multi-Model Training Summary', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        pivot_acc = summary_df.pivot(index='Target', columns='Algorithm', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=ax1, alpha=0.8)
        ax1.set_title('Test Accuracy by Algorithm and Target')
        ax1.set_ylabel('Accuracy')
        ax1.legend(title='Algorithm')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # F1-Score comparison
        pivot_f1 = summary_df.pivot(index='Target', columns='Algorithm', values='F1_Macro')
        pivot_f1.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Test F1-Score (Macro) by Algorithm and Target')
        ax2.set_ylabel('F1-Score (Macro)')
        ax2.legend(title='Algorithm')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        summary_path = os.path.join(self.results_dir, 'training_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: training_summary.png")
        
        # Save summary table
        summary_table_path = os.path.join(self.results_dir, 'training_summary.csv')
        summary_df.to_csv(summary_table_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved: training_summary.csv")
    
    def run_training_pipeline(self):
        """Execute complete training pipeline"""
        logger.info("EXECUTING STEP 1: MULTI-MODEL TRAINING PIPELINE")
        
        try:
            # Step 1: Load and preprocess data
            self.load_and_preprocess_data()
            
            # Step 2: Train all models
            self.train_all_models()
            
            # Step 3: Save results for evaluation
            self.save_training_results()
            
            logger.info("STEP 1 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("Multi-model baseline established")
            logger.info("All results saved for evaluation")
            logger.info("Ready for step1_evaluate.py")
            
            # Print results summary
            self._print_results_summary()
            
        except Exception as e:
            logger.error(f"STEP 1 TRAINING FAILED: {e}")
            raise
    
    def _print_results_summary(self):
        """Print summary of saved outputs"""
        print(f"\n📁 TRAINING RESULTS SUMMARY:")
        print(f"   All outputs saved to: {self.results_dir}")
        print(f"   🤖 Trained Models: {sum(len(m) for m in self.models.values())} models")
        for alg_name, models in self.models.items():
            print(f"      • {alg_name.upper()}: {len(models)} target models")
        print(f"   📊 Predictions:")
        print(f"      • all_predictions.csv")
        print(f"   📄 Metadata:")
        print(f"      • training_metadata.json")
        print(f"   📈 Summary:")
        print(f"      • training_summary.png/csv")
        print(f"\n➡️  Next: Run step1_evaluate.py for comprehensive evaluation!")


def get_training_config():
    """Configuration for Step 1 multi-model training"""
    return {
        'data_path': '../data/processed/credit_risk_dataset.csv',
        'exclude_columns': [
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
        
        # Algorithm-specific parameters (optional overrides)
        'xgboost_params': {
            # Uses defaults defined in train_all_models()
        },
        'mlp_params': {
            # Uses defaults defined in train_all_models()
        },
        'randomforest_params': {
            # Uses defaults defined in train_all_models()
        }
    }


# Main execution
if __name__ == "__main__":
    print("🚀 Starting Multi-Model Training - Step 1")
    print("="*60)
    
    # Get configuration
    config = get_training_config()
    
    # Create and run trainer
    trainer = MultiModelTrainer(config)
    trainer.run_training_pipeline()
    
    print("\n🏁 Step 1 Training execution completed!")
    print("Ready to run step1_evaluate.py for comprehensive evaluation!")
