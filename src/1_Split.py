"""
Strict Data Split for KSURE Risk Model
=====================================

Purpose: Implement strict data separation to prevent data leakage
- Train: for model training and feature selection
- Validation: for hyperparameter tuning
- OOT: for final unbiased evaluation

This ensures that:
1. Feature selection is done only on training data
2. Hyperparameter tuning uses train+validation
3. Final model uses train+validation
4. Grade rules are created from train+validation
5. Performance evaluation uses only OOT (never seen before)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_processed_data(data_path: str = "../data/processed/credit_risk_dataset_selected.csv") -> pd.DataFrame:
    """Load the processed dataset from 1_Dataset.py and add unique_id"""
    logger.info(f"Loading processed data from {data_path}")
    df = pd.read_csv(data_path)
    df['보험청약일자'] = pd.to_datetime(df['보험청약일자'])
    
    # 🔑 CREATE UNIQUE PRIMARY KEY - This eliminates all complex merge logic downstream
    df = df.reset_index(drop=True)  # Ensure clean index
    df['unique_id'] = df.index.astype(str)  # Create simple unique identifier
    
    logger.info(f"Loaded {len(df):,} records from {df['보험청약일자'].min()} to {df['보험청약일자'].max()}")
    logger.info(f"Added unique_id column (0 to {len(df)-1}) - This will simplify all downstream merging!")
    return df

def perform_strict_split(df: pd.DataFrame) -> dict:
    """
    Perform strict time-based data split
    
    Returns:
        dict with keys: 'train', 'validation', 'oot'
    """
    # Define split dates based on our analysis
    train_start = pd.to_datetime('2017-07-17')
    train_end = pd.to_datetime('2021-07-16')
    
    validation_start = pd.to_datetime('2021-07-17') 
    validation_end = pd.to_datetime('2023-01-16')
    
    oot_start = pd.to_datetime('2023-01-17')
    oot_end = pd.to_datetime('2024-07-16')
    
    # Create masks for each split
    train_mask = (df['보험청약일자'] >= train_start) & (df['보험청약일자'] <= train_end)
    validation_mask = (df['보험청약일자'] >= validation_start) & (df['보험청약일자'] <= validation_end)
    oot_mask = (df['보험청약일자'] >= oot_start) & (df['보험청약일자'] <= oot_end)
    
    # Create splits
    splits = {
        'train': df[train_mask].copy(),
        'validation': df[validation_mask].copy(),
        'oot': df[oot_mask].copy()
    }
    
    # Add split identifier to each dataset
    for split_name, split_df in splits.items():
        split_df['data_split'] = split_name
    
    return splits

def validate_splits(splits: dict) -> dict:
    """Validate the data splits and return statistics"""
    stats = {}
    total_records = sum(len(split_df) for split_df in splits.values())
    
    print("\n" + "="*60)
    print("📊 STRICT DATA SPLIT RESULTS")
    print("="*60)
    
    for split_name, split_df in splits.items():
        count = len(split_df)
        percentage = (count / total_records) * 100
        
        if count > 0:
            date_range = f"{split_df['보험청약일자'].min().strftime('%Y-%m-%d')} ~ {split_df['보험청약일자'].max().strftime('%Y-%m-%d')}"
            
            # Check default rate if risk_year4 exists
            default_rate = 0
            if 'risk_year4' in split_df.columns:
                default_rate = (split_df['risk_year4'] >= 1).mean()
            
            stats[split_name] = {
                'count': count,
                'percentage': percentage,
                'date_range': date_range,
                'default_rate': default_rate
            }
            
            print(f"\n🔹 {split_name.upper()} DATA:")
            print(f"   - Records: {count:,} ({percentage:.1f}%)")
            print(f"   - Period: {date_range}")
            print(f"   - Default rate: {default_rate:.1%}")
        else:
            print(f"\n⚠️  {split_name.upper()} DATA: No records found!")
    
    # Print usage strategy
    print(f"\n📋 USAGE STRATEGY:")
    train_val_count = len(splits['train']) + len(splits['validation'])
    print(f"   - Feature Selection (Step 2): TRAIN only = {len(splits['train']):,} records")
    print(f"   - Model Training (Steps 2-3): TRAIN + VALIDATION = {train_val_count:,} records")
    print(f"   - Grade Rules (Step 4): TRAIN + VALIDATION = {train_val_count:,} records")
    print(f"   - Final Evaluation (Step 5): OOT only = {len(splits['oot']):,} records")
    print(f"\n🔑 UNIQUE_ID BENEFITS:")
    print(f"   - All datasets now have 'unique_id' column for simple, reliable merging")
    print(f"   - No more complex composite key logic needed in Models 8, 9, 10")
    print(f"   - Eliminates row multiplication and duplicate handling issues")
    print("="*60)
    
    return stats

def save_splits(splits: dict, output_dir: str = "../data/splits") -> None:
    """Save the split datasets to separate files"""
    ensure_dir(output_dir)
    
    file_paths = {}
    for split_name, split_df in splits.items():
        file_path = os.path.join(output_dir, f"{split_name}_data.csv")
        split_df.to_csv(file_path, index=False)
        file_paths[split_name] = file_path
        logger.info(f"Saved {split_name} data: {file_path} ({len(split_df):,} records)")
    
    return file_paths

def save_split_metadata(stats: dict, file_paths: dict, output_dir: str = "../data/splits") -> None:
    """Save split metadata for tracking and reproducibility"""
    metadata = {
        'split_timestamp': datetime.now().isoformat(),
        'split_strategy': 'time_based_strict',
        'split_ratios': {
            'train': f"{stats['train']['percentage']:.1f}%",
            'validation': f"{stats['validation']['percentage']:.1f}%", 
            'oot': f"{stats['oot']['percentage']:.1f}%"
        },
        'split_details': stats,
        'file_paths': file_paths,
        'usage_rules': {
            'feature_selection': 'train_only',
            'hyperparameter_tuning': 'train_plus_validation',
            'final_model_training': 'train_plus_validation', 
            'grade_rule_creation': 'train_plus_validation',
            'performance_evaluation': 'oot_only'
        }
    }
    
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata: {metadata_path}")
    return metadata_path

def main() -> None:
    logger.info("Starting Strict Data Split Process")
    
    # Step 1: Load processed data from 1_Dataset.py
    df = load_processed_data()
    
    # Step 2: Perform strict time-based split
    logger.info("Performing strict time-based split...")
    splits = perform_strict_split(df)
    
    # Step 3: Validate splits and show statistics
    stats = validate_splits(splits)
    
    # Step 4: Save split datasets
    logger.info("Saving split datasets...")
    file_paths = save_splits(splits)
    
    # Step 5: Save metadata for reproducibility
    metadata_path = save_split_metadata(stats, file_paths)
    
    logger.info("Strict Data Split Completed!")
    logger.info(f"Output directory: ../data/splits/")
    logger.info(f"Metadata: {metadata_path}")
    logger.warning("IMPORTANT: All subsequent steps must respect these splits!")
    logger.warning("   - Never use OOT data for training, feature selection, or hyperparameter tuning")
    logger.warning("   - Only use OOT for final unbiased performance evaluation")

if __name__ == "__main__":
    main()
