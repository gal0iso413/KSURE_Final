#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Data Processing Script
Converts KED재무DATA.csv to KED가공재무DATA.csv

Process: KED재무DATA.csv -> 재무제표_process.py -> KED가공재무DATA.csv

This script processes financial data to create key financial ratios:
- ROE = 당기순이익 / 자기자본
- 영업이익률 = 영업이익 / 매출액  
- 부채비율 = 부채 / 자기자본
- 총자산회전율 = 매출액 / 자산
"""

import pandas as pd
import numpy as np
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class FinancialDataProcessor:
    """Process financial data to create financial ratios."""
    
    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Initialize the processor.
        
        Args:
            input_path: Path to the input CSV file (KED재무DATA.csv)
            output_path: Path to save the output CSV file (KED가공재무DATA.csv)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Column mapping for financial data
        self.cols = {
            'firm_id': '사업자등록번호',
            'date': '기준일자',
            'revenue': '매출액',
            'operating_income': '영업이익',
            'net_income': '당기순이익',
            'assets': '자산',
            'equity': '자기자본',
            'liabilities': '부채',
        }
        
        # Financial ratios to calculate
        self.ratios = {
            'ROE': 'ROE',
            '영업이익률': '영업이익률',
            '부채비율': '부채비율',
            '총자산회전율': '총자산회전율'
        }
    
    def validate_input_file(self) -> None:
        """Validate that the input file exists."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {self.input_path}")
        logger.info(f"Input file validated: {self.input_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load the financial data from CSV."""
        try:
            logger.info(f"Loading financial data from {self.input_path}")
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Validate that all required columns exist in the data."""
        logger.info("Validating required columns...")
        
        missing_cols = []
        for key, col in self.cols.items():
            if col not in df.columns:
                missing_cols.append(f"{col} (key: {key})")
        
        if missing_cols:
            error_msg = f"필수 컬럼 누락: {', '.join(missing_cols)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All required columns found")
        return True
    
    def convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert financial columns to numeric type."""
        logger.info("Converting financial columns to numeric...")
        
        numeric_cols = ['revenue', 'operating_income', 'net_income', 'assets', 'equity', 'liabilities']
        
        for key in numeric_cols:
            col = self.cols[key]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Converted {col} to numeric")
        
        return df
    
    def safe_division(self, numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
        """
        Perform safe division that handles NaN and zero values.
        
        Args:
            numerator: Numerator values
            denominator: Denominator values
            
        Returns:
            Division result with NaN for invalid cases
        """
        return np.where(
            (pd.notna(numerator)) & (pd.notna(denominator)) & (denominator != 0),
            numerator / denominator,
            np.nan
        )
    
    def calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial ratios from the data."""
        logger.info("Calculating financial ratios...")
        
        result = df.copy()
        
        # ROE = 당기순이익 / 자기자본
        result['ROE'] = self.safe_division(
            result[self.cols['net_income']], 
            result[self.cols['equity']]
        )
        logger.info("Calculated ROE")
        
        # 영업이익률 = 영업이익 / 매출액
        result['영업이익률'] = self.safe_division(
            result[self.cols['operating_income']], 
            result[self.cols['revenue']]
        )
        logger.info("Calculated 영업이익률")
        
        # 부채비율 = 부채 / 자기자본
        result['부채비율'] = self.safe_division(
            result[self.cols['liabilities']], 
            result[self.cols['equity']]
        )
        logger.info("Calculated 부채비율")
        
        # 총자산회전율 = 매출액 / 자산
        result['총자산회전율'] = self.safe_division(
            result[self.cols['revenue']], 
            result[self.cols['assets']]
        )
        logger.info("Calculated 총자산회전율")
        
        return result
    
    def select_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select only the required output columns."""
        logger.info("Selecting output columns...")
        
        # Keep only identifier columns and calculated ratios
        keep_cols = [
            self.cols['firm_id'], 
            self.cols['date'], 
            'ROE', '영업이익률', '부채비율', '총자산회전율'
        ]
        
        result = df[keep_cols]
        logger.info(f"Selected {len(keep_cols)} columns for output")
        
        return result
    
    def save_result(self, df: pd.DataFrame) -> None:
        """Save the processed data to CSV."""
        try:
            # Create output directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving result to {self.output_path}")
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Successfully saved {len(df)} rows with {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            raise
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for the calculated ratios."""
        logger.info("Generating summary statistics...")
        
        ratio_cols = ['ROE', '영업이익률', '부채비율', '총자산회전율']
        
        summary = {}
        for col in ratio_cols:
            if col in df.columns:
                summary[col] = {
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum()
                }
        
        return summary
    
    def process(self) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Execute the complete financial data processing pipeline."""
        logger.info("Starting financial data processing...")
        
        try:
            # Validate input file
            self.validate_input_file()
            
            # Load data
            df = self.load_data()
            
            # Validate columns
            self.validate_columns(df)
            
            # Convert numeric columns
            df = self.convert_numeric_columns(df)
            
            # Calculate financial ratios
            df = self.calculate_financial_ratios(df)
            
            # Select output columns
            result = self.select_output_columns(df)
            
            # Save result
            self.save_result(result)
            
            # Generate summary statistics
            summary = self.generate_summary_statistics(result)
            
            logger.info("Financial data processing completed successfully!")
            return result, summary
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main() -> None:
    """Main function to run the financial data processing."""
    # Configuration - using relative paths
    table_path = "../data/raw"
    input_file = "KED재무DATA.csv"
    output_file = "KED가공재무DATA.csv"
    
    input_path = os.path.join(table_path, input_file)
    output_path = os.path.join(table_path, output_file)
    
    logger.info("Starting Financial Data Processing")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    
    # Create processor and run
    processor = FinancialDataProcessor(input_path, output_path)
    result, summary = processor.process()
    
    # Log summary
    logger.info(f"Processing completed successfully")
    logger.info(f"Total records processed: {len(result)}")
    logger.info(f"Output columns: {list(result.columns)}")
    
    logger.info("Financial Ratios Summary:")
    for ratio, stats in summary.items():
        logger.info(f"{ratio}: Count={stats['count']:,}, Mean={stats['mean']:.4f}, "
                   f"Std={stats['std']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                   f"Null={stats['null_count']:,}")

if __name__ == "__main__":
    main()

