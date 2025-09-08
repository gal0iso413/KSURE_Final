# Import libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import logging
from typing import Optional, Dict, Any
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data collection libraries
try:
    from PublicDataReader import Ecos
    logger.info("Libraries imported successfully!")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please install missing packages using pip install commands above")
    raise

# Configuration
class Config:
    # API Key (Replace with your actual API key from https://ecos.bok.or.kr)
    ECOS_API_KEY = "X949JPUF94LO6EDI5EM5"  # Get from https://ecos.bok.or.kr
    
    # Output directory (relative path)
    OUTPUT_DIR = "../data/raw"
    
    # Default date range (can be overridden)
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2024, 12, 31)
    
    @classmethod
    def set_date_range(cls, start_date: str, end_date: str) -> None:
        """Set date range from string inputs."""
        try:
            cls.START_DATE = datetime.strptime(start_date, "%Y-%m-%d")
            cls.END_DATE = datetime.strptime(end_date, "%Y-%m-%d")
            logger.info(f"Date range set: {cls.START_DATE.strftime('%Y-%m-%d')} to {cls.END_DATE.strftime('%Y-%m-%d')}")
        except ValueError as e:
            logger.warning(f"Invalid date format: {e}. Using default range: 2020-01-01 to 2024-12-31")
            cls.START_DATE = datetime(2020, 1, 1)
            cls.END_DATE = datetime(2024, 12, 31)

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory: {Config.OUTPUT_DIR}")

class DataCollector:
    def __init__(self):
        """Initialize the data collector with ECOS API."""
        try:
            self.ecos_api = Ecos(Config.ECOS_API_KEY)
            logger.info("ECOS API initialized")
        except Exception as e:
            logger.error(f"ECOS API initialization error: {e}")
            raise ValueError("Valid ECOS API key required for production mode")
    
    def save_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """Save data to CSV file."""
        if data is not None and not data.empty:
            filepath = os.path.join(Config.OUTPUT_DIR, f"{filename}.csv")
            data.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"Saved: {filepath} ({len(data)} rows)")
            return True
        else:
            logger.warning(f"No data to save for {filename}")
            return False

# Initialize collector
collector = DataCollector()

def date_to_quarter(date: datetime) -> str:
    """Convert date to ECOS quarter format (YYYYQX)."""
    quarter = (date.month - 1) // 3 + 1
    return f"{date.year}Q{quarter}"

def collect_gdp_data() -> pd.DataFrame:
    """Collect GDP Growth Rate data with 2-track approach (current state + trend)."""
    logger.info("Starting GDP data collection (2-track: current + trend)")
    
    # Convert dates to quarter format
    start_quarter = date_to_quarter(Config.START_DATE)
    end_quarter = date_to_quarter(Config.END_DATE)
    
    logger.info(f"Period: {start_quarter} to {end_quarter}")
    
    # GDP Growth Rate YoY (Year-over-Year) - 10121
    logger.info("Collecting YoY growth rate...")
    gdp_yoy_raw = collector.ecos_api.get_statistic_search(
        통계표코드="200Y102",
        통계항목코드1="10121", 
        주기="Q",
        검색시작일자=start_quarter,
        검색종료일자=end_quarter
    )
    
    # GDP Growth Rate QoQ (Quarter-over-Quarter) - 10111
    logger.info("Collecting QoQ growth rate...")
    gdp_qoq_raw = collector.ecos_api.get_statistic_search(
        통계표코드="200Y102",
        통계항목코드1="10111", 
        주기="Q",
        검색시작일자=start_quarter,
        검색종료일자=end_quarter
    )
    
    if gdp_yoy_raw.empty or gdp_qoq_raw.empty:
        raise ValueError("No GDP data available from ECOS API")
    
    # Process YoY data
    quarter_periods_yoy = pd.PeriodIndex(gdp_yoy_raw['시점'], freq='Q')
    gdp_yoy_df = pd.DataFrame({
        'date': quarter_periods_yoy.to_timestamp(how='end'),
        'quarter': gdp_yoy_raw['시점'],
        'gdp_growth_rate_yoy': pd.to_numeric(gdp_yoy_raw['값'], errors='coerce')
    })
    
    # Process QoQ data
    quarter_periods_qoq = pd.PeriodIndex(gdp_qoq_raw['시점'], freq='Q')
    gdp_qoq_df = pd.DataFrame({
        'date': quarter_periods_qoq.to_timestamp(how='end'),
        'gdp_growth_rate_qoq': pd.to_numeric(gdp_qoq_raw['값'], errors='coerce')
    })
    
    # Merge YoY and QoQ data
    gdp_data = pd.merge(gdp_yoy_df, gdp_qoq_df, on='date', how='outer')
    gdp_data = gdp_data.sort_values('date').reset_index(drop=True)

    # Create 2-track GDP features: current state + trend
    # Track 1: Current quarterly growth rate (most recent value)
    gdp_data['GDP_성장률_분기'] = gdp_data['gdp_growth_rate_qoq']
    
    # Track 2: Annual trend (4-quarter moving average of YoY growth)
    gdp_data['GDP_성장률_연간추세'] = gdp_data['gdp_growth_rate_yoy'].rolling(window=4, min_periods=1).mean()
    
    # Keep only 2-track outputs
    gdp_data = gdp_data[['date', 'quarter', 'GDP_성장률_분기', 'GDP_성장률_연간추세']]

    logger.info(f"GDP data collected: {len(gdp_data)} quarters with 2-track features (current + trend)")
    
    # Change YYYY-MM-DD to YYYYMMDD
    gdp_data['date'] = gdp_data['date'].dt.strftime('%Y%m%d')

    collector.save_to_csv(gdp_data, 'gdp_data')
    return gdp_data


def collect_trade_data() -> pd.DataFrame:
    """Collect Export data with multiple frequency normal past-present change growth rates (MoM, QoQ, YoY)."""
    logger.info("Starting national trade indicators collection (normal past-present %)")
    
    start_month = f"{Config.START_DATE.year}{Config.START_DATE.month:02d}"
    end_month = f"{Config.END_DATE.year}{Config.END_DATE.month:02d}"
    logger.info(f"Collecting trade data from ECOS for period: {start_month} to {end_month}")
    
    # Export Values - USING KOREAN PARAMETER NAMES
    export_raw = collector.ecos_api.get_statistic_search(
        통계표코드="901Y011",
        통계항목코드1="FIEE",
        주기="M",
        검색시작일자=start_month,
        검색종료일자=end_month
    )
    
    if export_raw.empty:
        raise ValueError("No trade data available from ECOS API")
    
    # Create trade data
    # Convert '시점' (e.g., '202107') to the last day of the month
    export_periods = pd.PeriodIndex(export_raw['시점'], freq='M')
    export_df = pd.DataFrame({
        'date': export_periods.to_timestamp(how='end'),
        'export_value': pd.to_numeric(export_raw['값'], errors='coerce')
    })
    
    # Sort by date
    trade_data = export_df.sort_values('date')
    
    # Multi-frequency export growth rates (normal past-present change)
    # Formula: (current - past) / past
    trade_data['증감률_전년동기'] = (trade_data['export_value'] - trade_data['export_value'].shift(12)) / \
                                 trade_data['export_value'].shift(12)
    trade_data['증감률_전분기'] = (trade_data['export_value'] - trade_data['export_value'].shift(3)) / \
                                 trade_data['export_value'].shift(3)
    trade_data['증감률_전월'] = (trade_data['export_value'] - trade_data['export_value'].shift(1)) / \
                                 trade_data['export_value'].shift(1)
    
    # Keep normal past-present change features
    feature_columns = [
        '증감률_전년동기', '증감률_전분기', '증감률_전월'
    ]
    trade_data = trade_data[['date'] + feature_columns]
    trade_data = trade_data.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Trade data collected: {len(trade_data)} months with YoY/QoQ/MoM normal past-present change features")
    
    # Change YYYY-MM-DD to YYYYMMDD
    trade_data['date'] = trade_data['date'].dt.strftime('%Y%m%d')

    collector.save_to_csv(trade_data, 'trade_data')
    return trade_data


def collect_exchange_rate_data() -> pd.DataFrame:
    """Collect Exchange Rate data (USD, CNY) daily with rolling standard deviation volatility MoM, QoQ, YoY.
    MoM/QoQ/YoY volatility is computed as the natural coefficient of variation (std/mean)
    over the trailing 1/3/12 calendar months.
    """
    logger.info("Starting exchange rates collection (USD, CNY) - daily rolling std volatility")

    start_day = Config.START_DATE.strftime('%Y%m%d')
    end_day = Config.END_DATE.strftime('%Y%m%d')
    logger.info(f"Collecting exchange rates from ECOS for period: {start_day} to {end_day}")

    # ECOS: 731Y001 - Exchange rate table (daily)
    currency_item_codes = {
        '달러': '0000001',  # 미국
        '위안': '0000053',  # 중국
    }

    rate_level_columns = {}
    merged = None

    for cur_key, item_code in currency_item_codes.items():
        raw = collector.ecos_api.get_statistic_search(
            통계표코드="731Y001",
            통계항목코드1=item_code,
            주기="D",
            검색시작일자=start_day,
            검색종료일자=end_day
        )

        if raw is None or raw.empty:
            raise ValueError(f"No exchange rate data available for {cur_key.upper()} from ECOS API")

        # Determine time/value columns depending on ECOS endpoint variant
        time_col = '시점' if '시점' in raw.columns else ('TIME' if 'TIME' in raw.columns else None)
        value_col = '값' if '값' in raw.columns else ('DATA_VALUE' if 'DATA_VALUE' in raw.columns else None)
        if time_col is None or value_col is None:
            raise ValueError(f"Unexpected ECOS schema for {cur_key.upper()} exchange rate")

        # Parse daily date
        try:
            dates = pd.to_datetime(raw[time_col], format='%Y%m%d')
        except Exception:
            # Fallback: try generic parser
            dates = pd.to_datetime(raw[time_col])

        df = pd.DataFrame({
            'date': dates,
            f'{cur_key}_rate': pd.to_numeric(raw[value_col], errors='coerce')
        }).sort_values('date')

        rate_level_columns[cur_key] = f'{cur_key}_rate'
        merged = df if merged is None else pd.merge(merged, df, on='date', how='outer')

    fx = merged.sort_values('date').reset_index(drop=True)

    # Helper to compute rolling standard deviation volatility vs calendar offset
    def rolling_volatility_vs_offset(level_series: pd.Series, dates: pd.Series, offset: pd.DateOffset) -> pd.Series:
        # Combine dates and values into a single DataFrame with a DatetimeIndex for efficient slicing
        indexed_series = level_series.copy()
        indexed_series.index = dates
        
        results = []
        for current_date in dates:
            start_date = current_date - offset
            window_data = indexed_series.loc[start_date:current_date]

            if len(window_data) < 2:
                results.append(np.nan)
                continue
            
            # Calculate rolling standard deviation (coefficient of variation)
            # Formula: std / mean (natural coefficient of variation)
            if window_data.mean() == 0:
                volatility = np.nan  # Cannot compute when mean is zero
            else:
                volatility = window_data.std() / window_data.mean()
                results.append(volatility)
        
        # Create a new series with the original index to ensure alignment
        return pd.Series(results, index=level_series.index)

    # Compute MoM (1M), QoQ (3M), YoY (12M) rolling standard deviation volatility for each currency
    for cur_key, level_col in rate_level_columns.items():
        s = fx[level_col]
        d = fx['date']
        fx[f'{cur_key}_전월'] = rolling_volatility_vs_offset(s, d, pd.DateOffset(months=1))
        fx[f'{cur_key}_전분기'] = rolling_volatility_vs_offset(s, d, pd.DateOffset(months=3))
        fx[f'{cur_key}_전년동기'] = rolling_volatility_vs_offset(s, d, pd.DateOffset(years=1))

    # Keep only derived features and date (rolling standard deviation volatility only)
    feature_columns = [
        '달러_전년동기', '달러_전분기', '달러_전월',
        '위안_전년동기', '위안_전분기', '위안_전월',
    ]
    fx = fx[['date'] + feature_columns].sort_values('date').reset_index(drop=True)

    logger.info(f"Exchange rate data collected: {len(fx)} daily records with YoY/QoQ/MoM rolling standard deviation volatility features per currency")

    # Format date as YYYYMMDD
    fx['date'] = fx['date'].dt.strftime('%Y%m%d')

    collector.save_to_csv(fx, 'exchange_rate_data')
    return fx

def main(start_date: str = "2020-01-01", end_date: str = "2024-12-31") -> None:
    """Main data collection process."""
    logger.info("Starting Korean Economic Data Collection")
    logger.info("Focus: Maximum predictive power, minimum correlation")
    
    # Set date range
    Config.set_date_range(start_date, end_date)
    
    try:
        # Collect all datasets
        gdp_data = collect_gdp_data()
        trade_data = collect_trade_data()
        exchange_rate_data = collect_exchange_rate_data()

        logger.info("DATA COLLECTION SUMMARY")
        logger.info(f"GDP Growth Rate: {len(gdp_data):,} records (quarterly)")
        logger.info(f"National Trade Indicators: {len(trade_data):,} records (monthly)")
        logger.info(f"Exchange Rates: {len(exchange_rate_data):,} records (daily)")

        logger.info("FILES CREATED:")
        logger.info(f"   {Config.OUTPUT_DIR}/gdp_data.csv")
        logger.info(f"   {Config.OUTPUT_DIR}/trade_data.csv")
        logger.info(f"   {Config.OUTPUT_DIR}/exchange_rate_data.csv")

        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        logger.error("Please check:")
        logger.error("   1. ECOS API key is valid")
        logger.error("   2. Date range is appropriate")
        logger.error("   3. Network connection is stable")
        raise

if __name__ == "__main__":
    # Allow command line usage with optional date parameters
    import sys
    
    if len(sys.argv) == 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        main(start_date, end_date)
    else:
        # Use default dates
        main()
        
    logger.info("External data collection completed!")