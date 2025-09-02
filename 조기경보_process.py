#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Special Processing for Early Warning Data
Converts 조기경보내역.csv to 조기경보이력_리스크단계.csv

Process: 조기경보내역.sql -> special_processing.py -> 조기경보내역리스크단계.csv

This script processes early warning data to create a timeline of risk stages for each person.
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('special_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class EarlyWarningProcessor:
    """Process early warning data to create risk stage timeline."""
    
    def __init__(self, input_path, output_path, ongoing_end_date='2025-07-16'):
        """
        Initialize the processor.
        
        Args:
            input_path (str): Path to the input CSV file
            output_path (str): Path to save the output CSV file
            ongoing_end_date (str): Date to use for ongoing alerts
        """
        self.input_path = input_path
        self.output_path = output_path
        self.ongoing_end_date = pd.to_datetime(ongoing_end_date)
        
        # Risk level mapping based on reason codes
        self.reason_stage_mapping = {
            # LEVEL 0 (정상)
            '2': 0,  # 정상
            
            # LEVEL 1 (관심경보)
            '14': 1, # 단기연체이력
            '15': 1, # 행정처분
            '25': 1, # 배임횡령
            '26': 1, # 부정당제재
            '3': 1,  # 관심정보
            
            # LEVEL 2 (관찰경보) - 원래 2단계 경보이나, 리스크 단계상 LEVEL 1으로 병합
            '6': 1,  # 신용등급
            '5': 1,  # 재무정보
            '17': 1, # 상거래연체
            '18': 1, # 신용공여
            '22': 1, # 종업원
            '27': 1, # 대표자
            '21': 1, # 4대보험체납정보
            '20': 1, # 채무불이행이력
            '4': 1,  # 소송정보
            '7': 1,  # 기타정보
            
            # LEVEL 3 (연체경보) - 원래 3단계 경보이나, 리스크 단계상 LEVEL 2로 하향
            '8': 2,  # 단기연체
            
            # LEVEL 4 (위험경보) - 원래 4단계 경보이나, 리스크 단계상 LEVEL 3로 하향
            '23': 3, # 신용도판단
            '9': 3,  # 공공정보
            '13': 3, # 당좌부도
            '12': 3, # 휴폐업
            '11': 3, # 기업회생워크아웃
            '24': 3, # 청산해산
            '10': 3, # 채무불이행
        }
    
    def load_data(self):
        """Load the early warning data from CSV."""
        try:
            logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path, index_col=None)
            logger.info(f"Loaded {len(df)} records")
            return df
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data by converting date format."""
        logger.info("Preprocessing data...")
        
        # Convert date format
        df['조기경보발생일자'] = pd.to_datetime(df['조기경보발생일자'], format='%Y%m%d')
        
        # Sort by date
        df_sorted = df.sort_values('조기경보발생일자')
        
        logger.info(f"Data preprocessing completed. Date range: {df_sorted['조기경보발생일자'].min()} to {df_sorted['조기경보발생일자'].max()}")
        return df_sorted
    
    def process_individual_alerts(self, df_sorted):
        """Process individual alerts to determine start/end dates and risk stages."""
        logger.info("Processing individual alerts...")
        
        # Calculate termination dates for each alert group
        termination_dates = df_sorted[df_sorted['발생사유항목코드']==2].set_index(
            ['대상자번호','조기경보관리번호','동일조기경보그룹번호']
        )['조기경보발생일자']
        
        # Group active alarms (excluding termination codes)
        active_alarms_df = df_sorted[df_sorted['발생사유항목코드']!=2].copy()
        grouped = active_alarms_df.groupby(['대상자번호','조기경보관리번호','동일조기경보그룹번호'])
        
        # Aggregate alarm information
        alarms_df = grouped.agg(
            시작일=('조기경보발생일자','first'),
            발생사유항목코드=('발생사유항목코드','first')
        ).reset_index()
        
        # Map termination dates
        alarms_df['종료일'] = alarms_df.set_index(
            ['대상자번호','조기경보관리번호','동일조기경보그룹번호']
        ).index.map(termination_dates)
        
        # Fill missing termination dates with ongoing end date
        alarms_df['종료일'] = alarms_df['종료일'].fillna(self.ongoing_end_date)
        
        # Map risk stages
        alarms_df['리스크단계'] = alarms_df['발생사유항목코드'].astype(str).map(self.reason_stage_mapping)
        
        # Adjust end dates for same-day start/end
        is_same_day = alarms_df['시작일'] == alarms_df['종료일']
        alarms_df.loc[is_same_day, '종료일'] = alarms_df.loc[is_same_day, '종료일'] + pd.Timedelta(days=1)
        
        logger.info(f"Processed {len(alarms_df)} individual alerts")
        return alarms_df
    
    def create_risk_timeline(self, alarms_df):
        """Create a timeline of risk states for each person."""
        logger.info("Creating risk timeline...")
        
        # Extract all critical points (start and end dates)
        critical_points = []
        for _, row in alarms_df.iterrows():
            critical_points.append({'대상자번호': row['대상자번호'], '날짜': row['시작일']})
            critical_points.append({'대상자번호': row['대상자번호'], '날짜': row['종료일']})
        
        # Create timeline with unique dates
        timeline_df = pd.DataFrame(critical_points).drop_duplicates().sort_values(['대상자번호','날짜']).reset_index(drop=True)
        
        # Calculate risk state at each point
        timeline_states = []
        for _, point in timeline_df.iterrows():
            person_id, current_date = point['대상자번호'], point['날짜']
            
            # Find active alerts at current date
            active_alerts = alarms_df[
                (alarms_df['대상자번호'] == person_id) &
                (alarms_df['시작일'] <= current_date) & 
                (current_date < alarms_df['종료일'])
            ]
            
            if not active_alerts.empty:
                # Select highest risk alert
                max_risk_alert = active_alerts.loc[active_alerts['리스크단계'].idxmax()]
                timeline_states.append({
                    '날짜': current_date,
                    '대상자번호': person_id,
                    '리스크단계': max_risk_alert['리스크단계'],
                    '발생사유항목코드': max_risk_alert['발생사유항목코드']
                })
            else:
                # Normal state if no active alerts
                timeline_states.append({
                    '날짜': current_date,
                    '대상자번호': person_id,
                    '리스크단계': 0,
                    '발생사유항목코드': None
                })
        
        state_df = pd.DataFrame(timeline_states)
        logger.info(f"Created timeline with {len(state_df)} state changes")
        return state_df
    
    def merge_consecutive_periods(self, state_df):
        """Merge consecutive periods with the same risk level."""
        logger.info("Merging consecutive periods...")
        
        # Keep only points where risk level changes
        final_periods = state_df[state_df.groupby('대상자번호')['리스크단계'].transform(lambda x: x != x.shift())].copy()
        
        # Set end date as next state start date
        final_periods['종료일'] = final_periods.groupby('대상자번호')['날짜'].shift(-1)
        final_periods.rename(columns={'날짜':'시작일'}, inplace=True)
        
        # Filter out normal periods (risk level 0) and fill missing end dates
        final_result = final_periods[final_periods['리스크단계'] > 0].copy()
        final_result['종료일'] = final_result['종료일'].fillna(self.ongoing_end_date)
        
        # Convert dates and select final columns
        final_result['시작일'] = final_result['시작일'].dt.date
        final_result['종료일'] = final_result['종료일'].dt.date
        final_result = final_result[['대상자번호','리스크단계','발생사유항목코드','시작일','종료일']]
        final_result.sort_values(by=['대상자번호','시작일'], inplace=True)
        
        logger.info(f"Merged into {len(final_result)} final periods")
        return final_result
    
    def save_result(self, final_result):
        """Save the final result to CSV."""
        try:
            logger.info(f"Saving result to {self.output_path}")
            final_result.to_csv(self.output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Successfully saved {len(final_result)} records")
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            raise
    
    def process(self):
        """Execute the complete processing pipeline."""
        logger.info("Starting early warning data processing...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess data
            df_sorted = self.preprocess_data(df)
            
            # Process individual alerts
            alarms_df = self.process_individual_alerts(df_sorted)
            
            # Create risk timeline
            state_df = self.create_risk_timeline(alarms_df)
            
            # Merge consecutive periods
            final_result = self.merge_consecutive_periods(state_df)
            
            # Save result
            self.save_result(final_result)
            
            logger.info("Processing completed successfully!")
            return final_result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main():
    """Main function to run the processing."""
    # Configuration
    table_path = r"C:\Users\K_SURE_PROJECT\data_analysis\KSURE_Final\dataset"
    input_file = "조기경보내역.csv"
    output_file = "조기경보이력_리스크단계.csv"
    
    input_path = os.path.join(table_path, input_file)
    output_path = os.path.join(table_path, output_file)
    
    # Create processor and run
    processor = EarlyWarningProcessor(input_path, output_path)
    result = processor.process()
    
    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total records processed: {len(result)}")
    print(f"Unique persons: {result['대상자번호'].nunique()}")
    print(f"Risk level distribution:")
    print(result['리스크단계'].value_counts().sort_index())

if __name__ == "__main__":
    main()
