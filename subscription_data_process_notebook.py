# KED 청약 데이터 가공 노트북
# 
# 입력: `dataset/청약_backup.csv`
# 출력: `dataset/청약.csv`
# 
# 처리 사항:
# - '특별출연협약코드' 컬럼의 값을 확인하여 이진 변수 생성
# - 값이 17, 39, 또는 54이면 1, 그렇지 않으면(including NaN) 0
# - 새로운 컬럼명: '다이렉트보증여부'

# Cell 1: Import libraries and setup
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path('dataset/청약_backup.csv')
OUTPUT_PATH = Path('dataset/청약.csv')

print("📋 Subscription Data Processing Setup")
print("=" * 50)
print(f"Input: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Cell 2: Read and examine original data
print("\n📖 Reading original data...")
df = pd.read_csv(INPUT_PATH)
print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

# Cell 3: Examine '특별출연협약코드' column
print("\n📊 Examining '특별출연협약코드' column...")

# Check if column exists
if '특별출연협약코드' not in df.columns:
    raise ValueError("Column '특별출연협약코드' not found in the CSV file")

# Display unique values
unique_values = df['특별출연협약코드'].unique()
print(f"Unique values: {unique_values}")

# Count NaN values
nan_count = df['특별출연협약코드'].isna().sum()
print(f"NaN values: {nan_count:,} ({nan_count/len(df)*100:.2f}%)")

# Display value counts
print("\nValue counts:")
print(df['특별출연협약코드'].value_counts().head(10))

# Cell 4: Create binary column
print("\n🔧 Creating binary column '다이렉트보증여부'...")

# Convert to numeric, handling non-numeric values
df['특별출연협약코드'] = pd.to_numeric(df['특별출연협약코드'], errors='coerce')

# Create binary column: 1 if value is 17, 39, or 54; 0 otherwise (including NaN)
df['다이렉트보증여부'] = df['특별출연협약코드'].isin([17, 39, 54]).astype(int)

print("✅ Binary column created successfully!")

# Cell 5: Analyze results
print("\n📈 Analysis of results...")

# Count values
count_1 = df['다이렉트보증여부'].sum()
count_0 = (df['다이렉트보증여부'] == 0).sum()
total = len(df)

print(f"Total rows: {total:,}")
print(f"다이렉트보증여부 = 1: {count_1:,} ({count_1/total*100:.2f}%)")
print(f"다이렉트보증여부 = 0: {count_0:,} ({count_0/total*100:.2f}%)")

# Show examples of each category
print("\n📋 Examples of 다이렉트보증여부 = 1:")
sample_1 = df[df['다이렉트보증여부'] == 1][['사업자등록번호', '대상자명', '특별출연협약코드', '다이렉트보증여부']].head()
print(sample_1.to_string(index=False))

print("\n📋 Examples of 다이렉트보증여부 = 0:")
sample_0 = df[df['다이렉트보증여부'] == 0][['사업자등록번호', '대상자명', '특별출연협약코드', '다이렉트보증여부']].head()
print(sample_0.to_string(index=False))

# Cell 6: Data quality check
print("\n🔍 Data quality check...")

print("Missing values:")
print(df.isnull().sum())

print(f"\nUnique values in each column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique():,} unique values")

# Cell 7: Save processed data
print("\n💾 Saving processed data...")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ Saved: {OUTPUT_PATH}")

# Cell 8: Final verification
print("\n✅ Final verification...")
print(f"Final data shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")

# Verify the binary column logic
print(f"\n🔍 Verification of binary logic:")
print(f"  - Values 17, 39, 54 in original: {df['특별출연협약코드'].isin([17, 39, 54]).sum()}")
print(f"  - 다이렉트보증여부 = 1: {df['다이렉트보증여부'].sum()}")
print(f"  - Logic verification: {'✅ PASS' if df['특별출연협약코드'].isin([17, 39, 54]).sum() == df['다이렉트보증여부'].sum() else '❌ FAIL'}")

# Cell 9: Summary
print("\n🎉 Processing Summary")
print("=" * 50)
print(f"Input file: {INPUT_PATH}")
print(f"Output file: {OUTPUT_PATH}")
print(f"Total rows: {total:,}")
print(f"다이렉트보증여부 = 1: {count_1:,} ({count_1/total*100:.2f}%)")
print(f"다이렉트보증여부 = 0: {count_0:,} ({count_0/total*100:.2f}%)")
print(f"New column added: '다이렉트보증여부'")
print("\nProcessing completed successfully! 🚀")
