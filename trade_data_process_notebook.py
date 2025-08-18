# KED 무역 데이터 가공 노트북
# 
# 입력: `dataset/무역통계진흥원수출입실적.csv`
# 출력: `dataset/무역통계진흥원수출실적.csv`
# 
# 처리 사항:
# 1. 컬럼 삭제: '월별위탁가공미화금액값', '월별실적미화금액값'
# 2. 날짜 형식 변경: YYYYMM -> YYYYMMDD (DD는 해당 월의 마지막 날)
# 3. 중복 제거: 같은 사업자등록번호, 기준일자, 월별수출미화금액값을 가진 행

# Cell 1: Import libraries and setup
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import calendar

INPUT_PATH = Path('dataset/무역통계진흥원수출입실적.csv')
OUTPUT_PATH = Path('dataset/무역통계진흥원수출실적.csv')

print("📋 Trade Data Processing Setup")
print("=" * 50)
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Cell 2: Helper function for date conversion
def get_last_day_of_month(year_month_str):
    """
    Convert YYYYMM to YYYYMMDD where DD is the last day of the month
    Example: '202310' -> '20231031'
    """
    year = int(year_month_str[:4])
    month = int(year_month_str[4:6])
    last_day = calendar.monthrange(year, month)[1]
    return f"{year_month_str}{last_day:02d}"

print("✅ Helper function created: get_last_day_of_month()")

# Cell 3: Read and examine original data
print("\n📖 Reading original data...")
df = pd.read_csv(INPUT_PATH)
print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# Cell 4: Step 1 - Drop specified columns
print("\n🔧 Step 1: Dropping specified columns...")
columns_to_drop = ['월별위탁가공미화금액값', '월별실적미화금액값']
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Dropped columns: {columns_to_drop}")
print(f"Remaining columns: {list(df.columns)}")
print(f"Data shape after dropping columns: {df.shape}")

# Cell 5: Step 2 - Convert date format
print("\n📅 Step 2: Converting date format...")
print("Before conversion - sample dates:")
print(df['기준일자'].head())

df['기준일자'] = df['기준일자'].astype(str).apply(get_last_day_of_month)

print("\nAfter conversion - sample dates:")
print(df['기준일자'].head())
print("✅ Date format converted: YYYYMM -> YYYYMMDD")

# Cell 6: Step 3 - Remove duplicates
print("\n🧹 Step 3: Removing duplicates...")
original_count = len(df)
print(f"Original row count: {original_count:,}")

df = df.drop_duplicates(subset=['사업자등록번호', '기준일자', '월별수출미화금액값'])

final_count = len(df)
removed_count = original_count - final_count
print(f"Final row count: {final_count:,}")
print(f"Removed {removed_count:,} duplicate rows")
print(f"Reduction: {removed_count/original_count*100:.1f}%")

# Cell 7: Final data examination and save
print("\n📊 Final data examination...")
print(f"Final data shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

print("\nSample of processed data:")
print(df.head().to_string(index=False))

# Check for any remaining issues
print(f"\n🔍 Data quality check:")
print(f"Missing values:")
print(df.isnull().sum())

print(f"\nUnique values in each column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique():,} unique values")

# Cell 8: Save processed data
print("\n💾 Saving processed data...")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ Saved: {OUTPUT_PATH}")

# Cell 9: Summary
print("\n🎉 Processing Summary")
print("=" * 50)
print(f"Input file: {INPUT_PATH}")
print(f"Output file: {OUTPUT_PATH}")
print(f"Original rows: {original_count:,}")
print(f"Final rows: {final_count:,}")
print(f"Removed duplicates: {removed_count:,}")
print(f"Reduction rate: {removed_count/original_count*100:.1f}%")
print(f"Final columns: {len(df.columns)}")
print("\nProcessing completed successfully! 🚀")
