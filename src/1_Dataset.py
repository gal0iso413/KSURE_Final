"""
Credit Risk Dataset Creator
==========================

Creates time-series aware dataset combining:
- Insurance contracts (base table)
- Future risk outcomes (Y variables: Year1-4 risk predictions)
- Historical predictors (X variables: multiple data sources)

Design Principles:
- Temporal alignment prevents data leakage
- Mixed frequency handling via nearest past data
- Ordinal risk levels: 0 (no risk) → 1 → 2 → 3 (max risk)
- NaN for missing data periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetCreator:
    """Main class for creating credit risk prediction dataset"""
    
    def __init__(self, config: Dict):
        """
        Initialize with configuration
        
        Args:
            config: Dictionary containing:
                - base_table_path: Path to contracts table
                - risk_table_path: Path to risk outcomes table  
                - x_variable_paths: Dict of X variable file paths
                - lookback_periods: Dict of lookback periods by data type (monthly periods)
                - x_variable_modes: Dict of processing modes ('lookback' or 'nearest') by data type
                - x_aggregation_methods: Dict of aggregation methods by data type
                - prediction_horizons: List of years to predict [1,2,3,4]
                
        Note:
            Base table supports dual ID structure:
            - risk_id_column: for joining Y variables (risk outcomes)
            - firm_id_column: for joining X variables (predictors)
            
            X variable joining supports flexible two-column configuration:
            - join_columns: ('x_table_column', 'base_table_column') for different column names
            - join_columns: ('same_column', 'same_column') for same column names
            - No join_columns specified for market-level data
            
            X variable processing modes:
            - 'lookback': Creates multiple time periods information using time slice approach
                         Each period uses data from distinct, non-overlapping time windows
            - 'nearest': Finds N most recent data points within a fixed time window
            - 'static': Simple firm ID join without temporal processing (no date column needed)
        """
        self.config = config
        self.base_data = None
        self.final_dataset = None
        
        # Validate current_date configuration
        self._validate_current_date()
        
        logger.info("Dataset Creator Initialized")
        logger.info(f"Prediction horizons: {config.get('prediction_horizons', [1,2,3,4])} years")
        logger.info(f"X variable sources: {len(config.get('x_variable_paths', {}))} tables")
        logger.info(f"Current date: {config.get('current_date', '2025-07-16')}")
    
    def _validate_current_date(self):
        """Validate current_date configuration parameter"""
        current_date_config = self.config.get('current_date', '2025-07-16')
        
        try:
            if isinstance(current_date_config, str):
                # Validate string format
                datetime.strptime(current_date_config, '%Y-%m-%d')
            elif isinstance(current_date_config, datetime):
                # datetime object is valid
                pass
            elif hasattr(current_date_config, 'date'):
                # datetime-like object with date() method
                current_date_config.date()
            else:
                raise ValueError(f"Invalid current_date type: {type(current_date_config)}")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid current_date configuration '{current_date_config}': {e}. "
                           f"Expected format: 'YYYY-MM-DD' string or datetime object")
    
    def load_base_table(self) -> pd.DataFrame:
        """Load and prepare base contracts table"""
        logger.info("Loading Base Table")
        
        try:
            self.base_data = pd.read_csv(self.config['base_table_path'])
            logger.info(f"Base table loaded: {len(self.base_data):,} contracts")
            
            # Get column names from config
            base_columns = self.config['column_mappings']['base_table']
            date_col = base_columns['date_column']
            
            # Ensure date column is datetime
            self.base_data[date_col] = pd.to_datetime(self.base_data[date_col], format='%Y%m%d')
            
            # Add regime indicators for risk criteria change points
            change_point_1 = datetime(2019, 11, 12)
            change_point_2 = datetime(2021, 3, 4)
            
            # Categorical regime id: 0 = pre-2019-11-12, 1 = 2019-11-12..2021-03-03, 2 = 2021-03-04+
            mask_pre = self.base_data[date_col] < change_point_1
            mask_mid = (self.base_data[date_col] >= change_point_1) & (self.base_data[date_col] < change_point_2)
            mask_post = self.base_data[date_col] >= change_point_2

            # Use pandas nullable integer dtype to allow missing values if any
            regime_series = pd.Series(pd.NA, index=self.base_data.index, dtype='Int64')
            regime_series[mask_pre] = 0
            regime_series[mask_mid] = 1
            regime_series[mask_post] = 2
            self.base_data['조기경보선정기준변화'] = regime_series
            
            logger.info(f"Date column: {date_col}")
            logger.info(f"Date range: {self.base_data[date_col].min()} to {self.base_data[date_col].max()}")
            # Regime distribution summary
            regime_counts = self.base_data['조기경보선정기준변화'].value_counts().sort_index()
            pre_cnt = int(regime_counts.get(0, 0))
            mid_cnt = int(regime_counts.get(1, 0))
            post_cnt = int(regime_counts.get(2, 0))
            logger.info(f"Regime counts → pre_20191112: {pre_cnt}, 20191112_20210304: {mid_cnt}, post_20210304: {post_cnt}")
            return self.base_data
            
        except Exception as e:
            logger.error(f"Error loading base table: {e}")
            raise
    
    def join_y_variables(self) -> pd.DataFrame:
        """
        Join future risk outcomes for Year1-4 predictions
        
        NEW LOGIC (Individual Years):
        - For each contract, look at individual years (non-cumulative windows)
        - Year1: risks overlapping [contract_date, contract_date + 1 year]
        - Year2: risks overlapping [contract_date + 1 year, contract_date + 2 years]
        - Year3: risks overlapping [contract_date + 2 years, contract_date + 3 years]
        - Year4: risks overlapping [contract_date + 3 years, contract_date + 4 years]
        
        Risk Assignment Rules:
        - ALL risk levels (0,1,2,3) require COMPLETE prediction periods
        - NaN = prediction period not completed yet (insufficient time passed)
        - 0 = complete period with no risk events observed
        - 1,2,3 = complete period with risk events (max risk level observed)
        
        This ensures fair comparison and prevents temporal bias.
        """
        logger.info("Joining Y Variables (Future Risk)")
        
        try:
            # Load risk data
            risk_data = pd.read_csv(self.config['risk_table_path'])
            
            # Get column names from config
            risk_columns = self.config['column_mappings']['risk_table']
            base_columns = self.config['column_mappings']['base_table']
            
            risk_start_date_col = risk_columns['start_date_column']
            risk_end_date_col = risk_columns['end_date_column']
            risk_level_col = risk_columns['risk_level_column']
            risk_firm_id_col = risk_columns['risk_id_column']
            
            risk_data[risk_start_date_col] = pd.to_datetime(risk_data[risk_start_date_col], format='%Y-%m-%d')
            risk_data[risk_end_date_col] = pd.to_datetime(risk_data[risk_end_date_col], format='%Y-%m-%d')
            
            logger.info(f"Risk data loaded: {len(risk_data):,} risk events")
            logger.info(f"Risk columns: start_date={risk_start_date_col}, end_date={risk_end_date_col}, firm_id={risk_firm_id_col}, level={risk_level_col}")
            
            # Initialize result dataframe
            result_df = self.base_data.copy()
            contract_date_col = base_columns['date_column']
            # Use risk_id_column for Y variable joins (fallback to firm_id_column for backward compatibility)
            base_risk_id_col = base_columns.get('risk_id_column', base_columns['firm_id_column'])
            
            # Create Y variables for each prediction horizon
            prediction_horizons = self.config.get('prediction_horizons', [1, 2, 3, 4])
            
            # Get current date once for all processing
            current_date_config = self.config.get('current_date', '2025-07-16')
            if isinstance(current_date_config, str):
                current_date = datetime.strptime(current_date_config, '%Y-%m-%d').date()
            elif isinstance(current_date_config, datetime):
                current_date = current_date_config.date()
            else:
                current_date = current_date_config
            
            # Vectorized processing for all years
            for year in prediction_horizons:
                logger.info(f"Processing Year{year} risk...")
                
                # Vectorized calculation of prediction windows
                result_df[f'prediction_start_{year}'] = result_df[contract_date_col] + pd.DateOffset(years=(year-1))
                result_df[f'prediction_end_{year}'] = result_df[contract_date_col] + pd.DateOffset(years=year)
                
                # Check which prediction periods are complete
                prediction_end_dates = pd.to_datetime(result_df[f'prediction_end_{year}']).dt.date
                period_complete = prediction_end_dates <= current_date
                
                # Initialize risk column with NaN for incomplete periods
                result_df[f'risk_year{year}'] = np.where(period_complete, 0, np.nan)
                
                # Process only complete periods for risk calculation
                complete_contracts = result_df[period_complete].copy()
                
                if len(complete_contracts) > 0:
                    # Create risk mapping for efficient lookup
                    risk_mapping = {}
                    
                    # Group risk data by firm for faster lookup
                    risk_by_firm = risk_data.groupby(risk_firm_id_col)
                    
                    for firm_id, firm_risks in risk_by_firm:
                        risk_mapping[firm_id] = firm_risks
                    
                    # Process each complete contract
                    for idx in complete_contracts.index:
                        contract_row = complete_contracts.loc[idx]
                        risk_id = contract_row[base_risk_id_col]
                        prediction_start = contract_row[f'prediction_start_{year}']
                        prediction_end = contract_row[f'prediction_end_{year}']
                        
                        if risk_id in risk_mapping:
                            firm_risks = risk_mapping[risk_id]
                            
                            # Find overlapping risks using vectorized operations
                            overlapping_risks = firm_risks[
                                (firm_risks[risk_start_date_col] < prediction_end) &
                                (firm_risks[risk_end_date_col] > prediction_start)
                            ]
                            
                            if len(overlapping_risks) > 0:
                                max_risk = overlapping_risks[risk_level_col].max()
                                result_df.loc[idx, f'risk_year{year}'] = max_risk
                
                # Clean up temporary columns
                result_df.drop([f'prediction_start_{year}', f'prediction_end_{year}'], axis=1, inplace=True)
                
                # Calculate statistics
                year_risks = result_df[f'risk_year{year}']
                risk_count = ((year_risks > 0) & (~year_risks.isna())).sum()
                zero_count = (year_risks == 0).sum()
                nan_count = year_risks.isna().sum()
                
                logger.info(f"Year{year}: {risk_count} firms with risk, {zero_count} firms with no risk, {nan_count} missing data")
            
            self.base_data = result_df
            return result_df
            
        except Exception as e:
            logger.error(f"Error joining Y variables: {e}")
            raise
    
    def _parse_join_columns(self, x_columns: Dict, data_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse join columns configuration for X variable table
        
        Args:
            x_columns: Column configuration for X table
            data_type: Type of data (e.g., 'financial', 'macro')
            
        Returns:
            Tuple of (x_table_column, base_table_column) or (None, None) for market-level data
            
        Raises:
            ValueError: If join_columns configuration is invalid
        """
        # Check for new two-column format first
        if 'join_columns' in x_columns:
            join_config = x_columns['join_columns']
            
            if isinstance(join_config, (list, tuple)) and len(join_config) == 2:
                x_table_col, base_table_col = join_config
                return x_table_col, base_table_col
            elif isinstance(join_config, dict):
                x_table_col = join_config.get('x_table_column')
                base_table_col = join_config.get('base_table_column')
                if x_table_col is not None and base_table_col is not None:
                    return x_table_col, base_table_col
                else:
                    raise ValueError(f"Invalid join_columns dict for {data_type}: must contain 'x_table_column' and 'base_table_column'")
            else:
                raise ValueError(f"Invalid join_columns format for {data_type}: must be tuple/list of 2 elements or dict with 'x_table_column' and 'base_table_column'")
        
        # Fallback to old single-column format for backward compatibility
        elif 'firm_id_column' in x_columns:
            firm_id_col = x_columns['firm_id_column']
            if firm_id_col is None:
                return None, None  # Market-level data
            else:
                # Use same column name for both tables
                return firm_id_col, firm_id_col
        
        # No join columns specified - market-level data
        return None, None

    def _validate_join_columns(self, x_data: pd.DataFrame, x_table_join_col: str, base_table_join_col: str, data_type: str):
        """
        Validate that specified join columns exist in both tables
        
        Args:
            x_data: X variable dataframe
            x_table_join_col: Column name in X table for joining
            base_table_join_col: Column name in base table for joining
            data_type: Type of data for error messages
            
        Raises:
            ValueError: If join columns don't exist in respective tables
        """
        if x_table_join_col is not None:
            # Validate X table column exists
            if x_table_join_col not in x_data.columns:
                raise ValueError(f"Join column '{x_table_join_col}' not found in {data_type} table. Available columns: {list(x_data.columns)}")
            
            # Validate base table column exists
            if base_table_join_col not in self.base_data.columns:
                raise ValueError(f"Join column '{base_table_join_col}' not found in base table. Available columns: {list(self.base_data.columns)}")

    def join_x_variables(self) -> pd.DataFrame:
        """
        Join historical X variables from multiple sources
        
        Logic:
        - Three modes: 'lookback' (time slice intervals), 'nearest' (N most recent), or 'static' (no temporal processing)
        - Lookback mode uses non-cumulative time slices to prevent temporal leakage between periods
        - Data availability delays applied to ensure production-realistic modeling
        - Configurable aggregation methods (most_recent, mean, median, etc.) for temporal modes
        - Different lookback periods for different data types (monthly periods)
        - Static mode: Simple firm ID join without date filtering or time periods
        """
        logger.info("Joining X Variables (Historical Predictors)")
        
        result_df = self.base_data.copy()
        
        # Get base table column mappings
        base_columns = self.config['column_mappings']['base_table']
        contract_date_col = base_columns['date_column']
        base_firm_id_col = base_columns['firm_id_column']
        
        x_variable_paths = self.config.get('x_variable_paths', {})
        lookback_periods = self.config.get('lookback_periods', {})
        x_variable_modes = self.config.get('x_variable_modes', {})
        x_aggregation_methods = self.config.get('x_aggregation_methods', {})
        x_period_intervals = self.config.get('x_period_intervals', {})  # New: configurable intervals
        x_table_columns = self.config['column_mappings']['x_tables']
        
        for data_type, file_path in x_variable_paths.items():
            logger.info(f"Processing {data_type} data...")
            
            try:
                # Load X variable data
                x_data = pd.read_csv(file_path)
                
                # Get column mappings for this X table
                x_columns = x_table_columns.get(data_type, {})
                mode = x_variable_modes.get(data_type, 'lookback')  # Default 'lookback'
                
                # Handle date column based on mode
                if mode == 'static':
                    x_date_col = None  # Static mode doesn't need date column
                    logger.info(f"Static data: no date column required")
                else:
                    x_date_col = x_columns.get('date_column', 'date')
                    if x_date_col not in x_data.columns:
                        raise ValueError(f"Date column '{x_date_col}' not found in {data_type} table. Available columns: {list(x_data.columns)}")
                    x_data[x_date_col] = pd.to_datetime(x_data[x_date_col], format='%Y%m%d')
                    logger.info(f"Data range: {x_data[x_date_col].min()} to {x_data[x_date_col].max()}")
                
                x_table_join_col, base_table_join_col = self._parse_join_columns(x_columns, data_type)
                
                # Validate join columns exist in both tables
                self._validate_join_columns(x_data, x_table_join_col, base_table_join_col, data_type)
                
                # Get configuration for this data type (skip temporal config for static mode)
                if mode == 'static':
                    logger.info(f"Mode: {mode} - Static reference data")
                    logger.info(f"Firm-specific: {'Yes' if x_table_join_col else 'No (market-level)'}")
                    if x_table_join_col:
                        logger.info(f"Join: {x_table_join_col} (X table) ↔ {base_table_join_col} (base table)")
                else:
                    lookback = lookback_periods.get(data_type, 12)  # Default 12 periods
                    aggregation = x_aggregation_methods.get(data_type, 'most_recent')  # Default 'most_recent'
                    interval_days = self._get_period_interval_days(data_type, x_period_intervals)  # Get interval in days
                    
                    # Sort by date for efficient searching
                    x_data = x_data.sort_values(x_date_col)
                    
                    logger.info(f"Mode: {mode}, Periods: {lookback}, Interval: {interval_days} days, Aggregation: {aggregation}")
                    logger.info(f"Firm-specific: {'Yes' if x_table_join_col else 'No (market-level)'}")
                    if x_table_join_col:
                        logger.info(f"Join: {x_table_join_col} (X table) ↔ {base_table_join_col} (base table)")
                
                # Determine feature columns to use
                feature_cols = self._get_feature_columns(x_data, data_type, x_date_col, x_table_join_col)
                
                # Process based on mode
                if mode == 'lookback':
                    # Traditional lookback mode: create multiple time periods
                    self._process_lookback_mode_optimized(result_df, x_data, data_type, feature_cols, lookback, 
                                              contract_date_col, base_table_join_col, x_date_col, x_table_join_col, aggregation, interval_days)
                elif mode == 'nearest':
                    # Nearest mode: find N most recent data points within window
                    self._process_nearest_mode_optimized(result_df, x_data, data_type, feature_cols, lookback,
                                             contract_date_col, base_table_join_col, x_date_col, x_table_join_col, aggregation, interval_days)
                elif mode == 'static':
                    # Static mode: simple firm ID join without temporal processing
                    self._process_static_mode_optimized(result_df, x_data, data_type, feature_cols,
                                             base_table_join_col, x_table_join_col)
                else:
                    logger.warning(f"Unknown mode '{mode}' for {data_type}, skipping...")
                    continue
                
                feature_count = len([col for col in result_df.columns if col.startswith(f"{data_type}_")])
                if mode == 'static':
                    logger.info(f"Added {feature_count} features from {data_type} ({len(feature_cols)} static columns)")
                else:
                    logger.info(f"Added {feature_count} features from {data_type} ({len(feature_cols)} columns × {lookback} periods)")
                
            except Exception as e:
                logger.error(f"Error processing {data_type}: {e}")
                continue
        
        self.final_dataset = result_df
        return result_df
    
    def _get_feature_columns(self, x_data: pd.DataFrame, data_type: str, x_date_col: Optional[str], x_table_join_col: str) -> List[str]:
        """
        Determine which columns to use as features from X variable table
        
        Logic:
        1. Start with all columns
        2. Remove system columns (date, firm_id)
        3. Apply include filter (if specified)
        4. Apply exclude filter
        
        Args:
            x_data: The X variable dataframe
            data_type: Type of data (e.g., 'financial', 'macro')
            x_date_col: Date column name
            x_firm_id_col: Firm ID column name (or None for market-level data)
            
        Returns:
            List of column names to use as features
        """
        # Start with all columns
        all_columns = list(x_data.columns)
        
        # Remove system columns (date and firm_id)
        system_cols = []
        if x_date_col is not None:
            system_cols.append(x_date_col)
        if x_table_join_col is not None:
            system_cols.append(x_table_join_col)
        
        # Apply include filter first (if specified)
        include_columns = self.config.get('x_include_columns', {}).get(data_type, None)
        if include_columns is not None:
            # Only keep specified columns (that exist in the data)
            feature_cols = [col for col in include_columns if col in all_columns]
            logger.info(f"Include filter: {len(feature_cols)}/{len(include_columns)} columns found")
        else:
            # Use all columns except system columns
            feature_cols = [col for col in all_columns if col not in system_cols]
        
        # Apply exclude filter
        exclude_columns = self.config.get('x_exclude_columns', {}).get(data_type, [])
        if exclude_columns:
            before_count = len(feature_cols)
            feature_cols = [col for col in feature_cols if col not in exclude_columns]
            excluded_count = before_count - len(feature_cols)
            if excluded_count > 0:
                logger.info(f"Exclude filter: removed {excluded_count} columns")
        
        return feature_cols
    
    def _get_period_interval_days(self, data_type: str, x_period_intervals: Dict) -> int:
        """
        Get period interval in days for a data type
        
        Args:
            data_type: Type of data (e.g., 'financial', 'macro')
            x_period_intervals: Configuration dict with interval specifications
            
        Returns:
            Number of days between periods
        """
        interval_config = x_period_intervals.get(data_type, 'monthly')

        mapping = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'yearly': 365,
        }

        # 1) Direct integer-like values (including numpy integer types)
        if isinstance(interval_config, (int, np.integer)):
            return max(1, int(interval_config))

        # 2) String values: try known keywords, then numeric strings
        if isinstance(interval_config, str):
            s = interval_config.strip().lower()
            if s in mapping:
                return mapping[s]
            try:
                # Accept numeric strings like "730" or "365.0"
                days = int(float(s))
                return max(1, days)
            except ValueError:
                pass

        # 3) Fallback
        logger.warning(f"Unknown interval '{interval_config}' for {data_type}, using monthly (30 days)")
        return 30
    
    def _apply_availability_delay(self, contract_date: datetime, data_type: str) -> datetime:
        """
        Apply real-world data availability delays to contract date
        
        Args:
            contract_date: Original contract date
            data_type: Type of data (e.g., 'financial', 'gdp')
            
        Returns:
            Effective contract date adjusted for data availability delays
            
        Example:
            If financial data has 45-day delay and safety margin is 7 days,
            a contract on 2024-01-01 becomes effective 2023-11-09 for financial data lookback
        """
        # Get data availability delays configuration
        data_delays = self.config.get('data_availability_delays', {})
        apply_delays = self.config.get('apply_availability_delays', False)
        
        if not apply_delays:
            return contract_date  # No delays applied
        
        # Get delay for this data type (default 0 if not specified)
        base_delay = data_delays.get(data_type, 0)
        
        # Add safety margin if configured
        safety_margin = self.config.get('safety_margin_days', 0)
        total_delay = base_delay + safety_margin
        
        # Apply delay by moving contract date backward
        effective_date = contract_date - timedelta(days=total_delay)
        
        return effective_date
    
    def _process_lookback_mode_optimized(self, result_df: pd.DataFrame, x_data: pd.DataFrame, data_type: str, 
                              feature_cols: List[str], lookback: int, contract_date_col: str, 
                              base_table_join_col: str, x_date_col: str, x_table_join_col: str, aggregation: str, interval_days: int):
        """
        Lookback mode with time slice approach and change-rate features

        Time slices (non-cumulative):
        - t0: [contract_date - interval_days, contract_date]
        - t1: (contract_date - 2*interval_days, contract_date - interval_days]
        - t2: (contract_date - 3*interval_days, contract_date - 2*interval_days]

        Feature outputs:
        - Raw level for t0 only: {data_type}_{feature}_t0
        - Change-rate for later slices vs t0: {data_type}_{feature}_변화율_{k}{unit}, k=1..lookback-1
          where unit in {y,q,m,w,d} according to the configured interval
        """
        # Determine unit label used in 변화율 column names
        unit_label = self._get_interval_unit_label(data_type, self.config.get('x_period_intervals', {}))

        # Pre-initialize columns: t0 raw + change-rate columns for k>=1
        for col in feature_cols:
            t0_name = f"{data_type}_{col}_t0"
            if t0_name not in result_df.columns:
                result_df[t0_name] = np.nan
            # Initialize 변화율 columns for k = 1..lookback-1
            for k in range(1, max(lookback, 1)):
                rate_col = f"{data_type}_{col}_변화율_{k}{unit_label}"
                if rate_col not in result_df.columns:
                    result_df[rate_col] = np.nan
        
        # Get unique contract dates and firm IDs to optimize queries
        contract_dates = result_df[contract_date_col].unique()
        firm_ids = None
        if x_table_join_col is not None:
            firm_ids = result_df[base_table_join_col].unique()
        
        # Process by contract date to minimize data filtering
        for contract_date in contract_dates:
            # Apply data availability delay if configured
            effective_contract_date = self._apply_availability_delay(contract_date, data_type)
            
            # Get all contracts for this date
            date_mask = result_df[contract_date_col] == contract_date
            date_contracts = result_df[date_mask]
            
            # Calculate date windows for all periods at once
            min_search_date = effective_contract_date - timedelta(days=interval_days * lookback)
            
            # Pre-filter x_data by date range (single filter operation)
            date_filtered_data = x_data[
                (x_data[x_date_col] <= effective_contract_date) & 
                (x_data[x_date_col] >= min_search_date)
            ]
            
            if len(date_filtered_data) == 0:
                continue  # No data available for this date range
            
            # Process each contract for this date
            for contract_idx in date_contracts.index:
                contract_firm_id = result_df.loc[contract_idx, base_table_join_col]
                
                # Filter by firm if needed (once per contract)
                if x_table_join_col is not None:
                    firm_data = date_filtered_data[date_filtered_data[x_table_join_col] == contract_firm_id]
                else:
                    firm_data = date_filtered_data
                
                if len(firm_data) == 0:
                    continue  # No data for this firm
                
                # Compute aggregated values for all periods first (t0..t{lookback-1})
                period_to_values: Dict[int, Dict[str, float]] = {}
                for period in range(lookback):
                    period_end_date = effective_contract_date - timedelta(days=interval_days * period)
                    period_start_date = effective_contract_date - timedelta(days=interval_days * (period + 1))
                    period_data = firm_data[
                        (firm_data[x_date_col] <= period_end_date) &
                        (firm_data[x_date_col] > period_start_date)
                    ]
                    period_to_values[period] = self._aggregate_data(period_data, feature_cols, aggregation)

                # Assign t0 raw
                t0_values = period_to_values.get(0, {})
                for col in feature_cols:
                    t0_name = f"{data_type}_{col}_t0"
                    result_df.loc[contract_idx, t0_name] = t0_values.get(col, np.nan)

                # Assign change-rate for k>=1 vs t0
                for k in range(1, lookback):
                    tk_values = period_to_values.get(k, {})
                    for col in feature_cols:
                        current_val = t0_values.get(col, np.nan)
                        past_val = tk_values.get(col, np.nan)
                        
                        # Compute symmetric percentage change: (current - past) / ((|current| + |past|) / 2)
                        # This formula prevents division by zero issues and handles negative values gracefully
                        # It provides symmetric treatment: change from A to B equals negative change from B to A
                        if pd.isna(current_val) or pd.isna(past_val):
                            rate = np.nan
                        elif current_val == 0 and past_val == 0:
                            rate = 0.0  # No change when both values are zero
                        else:
                            # Symmetric percentage change formula - robust against zero denominators and sign changes
                            rate = (current_val - past_val) / ((abs(current_val) + abs(past_val)) / 2)
                        
                        rate_col = f"{data_type}_{col}_변화율_{k}{unit_label}"
                        result_df.loc[contract_idx, rate_col] = rate

    def _get_interval_unit_label(self, data_type: str, x_period_intervals: Dict) -> str:
        """
        Map interval configuration to a short unit label for column names.
        - yearly -> 'y', quarterly -> 'q', monthly -> 'm', weekly -> 'w', daily -> 'd'
        - integer days -> 'd'
        """
        interval_config = x_period_intervals.get(data_type, 'monthly')
        if isinstance(interval_config, int):
            return 'd'
        mapping = {
            'daily': 'd',
            'weekly': 'w',
            'monthly': 'm',
            'quarterly': 'q',
            'yearly': 'y',
        }
        return mapping.get(interval_config, 'm')
    
    def _process_nearest_mode_optimized(self, result_df: pd.DataFrame, x_data: pd.DataFrame, data_type: str,
                             feature_cols: List[str], periods: int, contract_date_col: str,
                             base_table_join_col: str, x_date_col: str, x_table_join_col: str, aggregation: str, interval_days: int):
        """
        Optimized nearest mode processing: find N nearest data points within a single time period
        
        Logic:
        - interval_days: defines the single past time period to search within (fixed window)
        - periods: number of nearest data points to find within that interval
        - Example: interval_days=90, periods=1 means "find 1 nearest point within past 90 days"
        """
        # Pre-initialize all feature columns
        for period in range(periods):
            for col in feature_cols:
                feature_name = f"{data_type}_{col}_{period}"
                if feature_name not in result_df.columns:
                    result_df[feature_name] = np.nan
        
        # Get unique contract dates for optimization
        contract_dates = result_df[contract_date_col].unique()
        
        # Process by contract date to minimize data filtering
        for contract_date in contract_dates:
            # Apply data availability delay if configured
            effective_contract_date = self._apply_availability_delay(contract_date, data_type)
            
            # Get all contracts for this date
            date_mask = result_df[contract_date_col] == contract_date
            date_contracts = result_df[date_mask]
            
            # Define search window: fixed interval_days back (NOT multiplied by periods)
            min_search_date = effective_contract_date - timedelta(days=interval_days)
            
            # Pre-filter x_data by date range (single filter operation)
            date_filtered_data = x_data[
                (x_data[x_date_col] <= effective_contract_date) & 
                (x_data[x_date_col] >= min_search_date)
            ]
            
            if len(date_filtered_data) == 0:
                continue  # No data available for this date range
            
            # Sort once for this date range (more efficient than sorting per contract)
            date_filtered_data = date_filtered_data.sort_values(x_date_col, ascending=False)
            
            # Handle market-level data efficiently (same for all contracts)
            if x_table_join_col is None:
                # Market-level data: find N most recent records once and apply to all contracts
                recent_records = date_filtered_data.head(periods)
                
                # Apply to all contracts for this date
                for contract_idx in date_contracts.index:
                    for period in range(periods):
                        if period < len(recent_records):
                            record = recent_records.iloc[period]
                            for col in feature_cols:
                                feature_name = f"{data_type}_{col}_{period}"
                                result_df.loc[contract_idx, feature_name] = record[col]
            else:
                # Firm-specific data: process each contract individually
                for contract_idx in date_contracts.index:
                    contract_firm_id = result_df.loc[contract_idx, base_table_join_col]
                    
                    # Filter by firm
                    firm_data = date_filtered_data[date_filtered_data[x_table_join_col] == contract_firm_id]
                    
                    # Take the N most recent records within the fixed interval
                    recent_records = firm_data.head(periods)
                    
                    # Create features for each of the N most recent records
                    for period in range(periods):
                        if period < len(recent_records):
                            # Use specific record
                            record = recent_records.iloc[period]
                            for col in feature_cols:
                                feature_name = f"{data_type}_{col}_{period}"
                                result_df.loc[contract_idx, feature_name] = record[col]
                                             # If no more data available, NaN values are already set during initialization
    
    def _process_static_mode_optimized(self, result_df: pd.DataFrame, x_data: pd.DataFrame, data_type: str,
                             feature_cols: List[str], base_table_join_col: str, x_table_join_col: str):
        """
        Optimized static mode processing: simple firm ID join without temporal processing
        
        Logic:
        - No date filtering or time periods
        - Simple join based on firm ID columns
        - One feature per column (no time suffixes)
        - Direct value assignment without aggregation
        """
        # Pre-initialize all feature columns
        for col in feature_cols:
            feature_name = f"{data_type}_{col}"
            if feature_name not in result_df.columns:
                result_df[feature_name] = np.nan
        
        # Process based on whether firm-specific or market-level
        if x_table_join_col is not None:
            # Firm-specific static data
            logger.info(f"Processing firm-specific static data...")
            
            # Create a mapping from firm ID to feature values for efficient lookup
            firm_mapping = {}
            for _, row in x_data.iterrows():
                firm_id = row[x_table_join_col]
                if firm_id not in firm_mapping:
                    firm_mapping[firm_id] = {}
                
                # Store all feature values for this firm
                for col in feature_cols:
                    if col in x_data.columns:
                        firm_mapping[firm_id][col] = row[col]
            
            # Apply mapping to result dataframe
            for idx, row in result_df.iterrows():
                contract_firm_id_raw = row[base_table_join_col]
                
                # Skip join if base table has missing value
                if pd.isna(contract_firm_id_raw):
                    continue  # Keep pre-initialized NaN values
                
                # Convert float to string for matching (25933.0 → "25933")
                if isinstance(contract_firm_id_raw, float):
                    contract_firm_id = str(int(contract_firm_id_raw))
                else:
                    contract_firm_id = str(contract_firm_id_raw)
                
                # Skip join if no matching value found in industry table
                if contract_firm_id not in firm_mapping:
                    continue  # Keep pre-initialized NaN values
                
                # Apply industry data for matching firms
                firm_data = firm_mapping[contract_firm_id]
                for col in feature_cols:
                    feature_name = f"{data_type}_{col}"
                    result_df.loc[idx, feature_name] = firm_data.get(col, np.nan)
        else:
            # Market-level static data (same values for all contracts)
            logger.info(f"Processing market-level static data...")
            
            if len(x_data) > 0:
                # Use the first row (assuming all rows have same values for market-level data)
                market_data = x_data.iloc[0]
                for col in feature_cols:
                    feature_name = f"{data_type}_{col}"
                    if col in x_data.columns:
                        result_df[feature_name] = market_data[col]
                    else:
                        result_df[feature_name] = np.nan
    
    def _aggregate_data(self, data: pd.DataFrame, feature_cols: List[str], aggregation: str) -> Dict:
        """
        Aggregate data based on specified method
        """
        if len(data) == 0:
            return {col: np.nan for col in feature_cols}
        
        result = {}
        for col in feature_cols:
            if col not in data.columns:
                result[col] = np.nan
                continue
                
            if aggregation == 'most_recent':
                result[col] = data.iloc[-1][col]  # Most recent (data is sorted by date)
            elif aggregation == 'mean':
                result[col] = data[col].mean()
            elif aggregation == 'median':
                result[col] = data[col].median()
            elif aggregation == 'max':
                result[col] = data[col].max()
            elif aggregation == 'min':
                result[col] = data[col].min()
            else:
                logger.warning(f"Unknown aggregation method '{aggregation}', using most_recent")
                result[col] = data.iloc[-1][col]
        
        return result
    
    def create_dataset(self) -> pd.DataFrame:
        """
        Main method to create complete dataset
        
        Returns:
            Complete dataset with Y and X variables
        """
        logger.info("Creating Credit Risk Dataset")
        
        # Step 1: Load base table
        self.load_base_table()
        
        # Step 2: Join Y variables (future risk)
        self.join_y_variables()
        
        # Step 3: Join X variables (historical predictors)
        self.join_x_variables()
        
        # Final summary
        self.print_dataset_summary()
        
        return self.final_dataset
    
    def print_dataset_summary(self) -> None:
        """Print comprehensive dataset summary"""
        if self.final_dataset is None:
            logger.error("No dataset created yet")
            return
        
        logger.info("DATASET SUMMARY")
        
        df = self.final_dataset
        
        # Basic stats
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total features: {len(df.columns):,}")
        
        # Regime summary
        if '조기경보선정기준변화' in df.columns:
            regime_counts = df['조기경보선정기준변화'].value_counts().sort_index()
            logger.info(f"조기경보선정기준변화 (0=pre-20191112, 1=20191112-20210303, 2=post-20210304): {dict(regime_counts)}")
        
        # Y variable summary
        y_cols = [col for col in df.columns if col.startswith('risk_year')]
        logger.info(f"Y Variables: {len(y_cols)}")
        for col in y_cols:
            risk_dist = df[col].value_counts().sort_index()
            nan_count = df[col].isna().sum()
            logger.info(f"   {col}: {dict(risk_dist)} (NaN: {nan_count})")
        
        # X variable summary by type
        x_data_types = set()
        for col in df.columns:
            # Include t0 features and 변화율 features
            if not col.startswith('risk_year') and ("_t" in col or "_변화율_" in col):
                data_type = col.split('_')[0]
                x_data_types.add(data_type)
        
        logger.info(f"X Variable Types: {len(x_data_types)}")
        for data_type in sorted(x_data_types):
            type_cols = [col for col in df.columns if col.startswith(f"{data_type}_")]
            logger.info(f"   {data_type}: {len(type_cols)} features")
        
        # Missing data summary
        logger.info("Missing Data Summary:")
        missing_pct = (df.isna().sum() / len(df) * 100).round(1)
        high_missing = missing_pct[missing_pct > 10].sort_values(ascending=False)
        if len(high_missing) > 0:
            logger.info(f"   Features with >10% missing: {len(high_missing)}")
            for col, pct in high_missing.head(20).items():
                logger.info(f"      {col}: {pct}%")
        else:
            logger.info("   All features have <10% missing data")
        
        logger.info("Dataset creation completed!")
    
    def save_dataset(self, output_path: str) -> None:
        """Save final dataset to CSV"""
        if self.final_dataset is None:
            logger.error("No dataset to save")
            return
        
        self.final_dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to: {output_path}")
        logger.info(f"Shape: {self.final_dataset.shape}")


# Example configuration
def get_example_config():
    """Example configuration for dataset creation"""
    return {
        # File paths
        'base_table_path': '../data/raw/청약.csv',
        'risk_table_path': '../data/processed/조기경보이력_리스크단계.csv',
        'x_variable_paths': {
            '재무정보': '../data/processed/KED가공재무DATA.csv',
            '수출실적': '../data/raw/무역통계진흥원수출입실적.csv',
            '신용등급': '../data/raw/KED종합신용정보.csv', 
            'GDP': '../data/raw/gdp_data.csv',
            '총수출': '../data/raw/trade_data.csv',
            '업종': '../data/raw/업종코드_수출자.csv',
            '환변동': '../data/raw/exchange_rate_data.csv',
        },
        
        # Table-specific column mappings
        'column_mappings': {
            'base_table': {
                'date_column': '보험청약일자',
                'risk_id_column': '수출자대상자번호',  # ID for joining Y variables (risk outcomes)
                'firm_id_column': '사업자등록번호'     # ID for joining X variables (predictors)
            },
            'risk_table': {
                'start_date_column': '시작일',
                'end_date_column': '종료일',
                'risk_id_column': '대상자번호',  # Should match risk_id_column from base_table
                'risk_level_column': '리스크단계'
            },
            'x_tables': {
                '재무정보': {
                    'date_column': '기준일자',
                    'join_columns': ('사업자등록번호', '사업자등록번호')  # (x_table_column, base_table_column)
                },
                '수출실적': {
                    'date_column': '기준일자',
                    'join_columns': ('사업자등록번호', '사업자등록번호')  # (x_table_column, base_table_column)
                },
                '신용등급': {
                    'date_column': '평가일자',
                    'join_columns': ('사업자등록번호', '사업자등록번호')  # (x_table_column, base_table_column)
                },
                'GDP': {
                    'date_column': 'date', 
                    # No join_columns specified = market-level data
                },
                '총수출': {
                    'date_column': 'date', 
                    # No join_columns specified = market-level data
                },
                '업종': {
                    'join_columns': ('업종코드', '업종코드1')  # (x_table_column, base_table_column)
                    # No date_column needed for static mode
                },
                '환변동': {
                    'date_column': 'date', 
                    # No join_columns specified = market-level data
                }
            }
        },
        
        # Specific columns to include from X variables (if not specified, uses all columns)
        'x_include_columns': {
            '재무정보': None,
            '총수출': None,
            '신용등급': None, 
            'GDP': None,
            '총수출': None,
            '업종': None,
            '환변동': None,
        },
        
        # Columns to exclude from X variables (applied after include filter)
        'x_exclude_columns': {
            '재무정보': [],
            '수출실적': [],
            '신용등급': [],
            'GDP': ['quarter'],
            '총수출': [],
            '업종': ['중분류','세세분류'],
            '환변동': [],
        },
        
        # Lookback periods for each data type
        'lookback_periods': {
            '재무정보': 3,  # 3 periods of financial data
            '수출실적': 3,
            '신용등급': 1,       # 1 period of grade data
            'GDP': 1,
            '총수출': 1,
            '업종': 1,
            '환변동': 1,
        },
        
        # Period intervals (how many days between periods)
        'x_period_intervals': {
            '재무정보': 'yearly',  # 365 days for quarterly financial reports
            '수출실적': 'yearly',
            '신용등급': 'yearly',
            'GDP': 'yearly',
            '총수출': 'yearly',
            '업종': 'yearly',
            '환변동': 'yearly',
            # Options: 'daily'(1), 'weekly'(7), 'monthly'(30), 'quarterly'(90), 'yearly'(365), or integer days
        },
        
        # X variable processing modes
        'x_variable_modes': {
            '재무정보': 'lookback',  # Creates: financial_revenue_t0, financial_revenue_t1, ... 
            '수출실적': 'lookback',
            '신용등급': 'nearest',       # Creates: grade_score_0 (most recent grade)
            'GDP': 'nearest',
            '총수출': 'nearest',
            '업종': 'static',     # Creates: industry_코드, industry_분류명 (no time suffixes)
            '환변동': 'nearest',
        },
        

        
        # Aggregation methods for X variables (not used for static mode)
        'x_aggregation_methods': {
            '재무정보': 'mean',      # Average financial data in each period (smooths outliers)
            '수출실적': 'mean',
            '신용등급': 'most_recent',   # Most recent grade in each period
            'GDP': 'mean',
            '총수출': 'mean',    
            # '업종': not needed for static mode
            '환변동': 'mean',
        },
        
        # Prediction horizons
        'prediction_horizons': [1, 2, 3, 4],  # Years to predict
        
        # Current date for dataset creation (ensures reproducibility)
        'current_date': '2025-07-16',  # Format: 'YYYY-MM-DD' or datetime object
        
        # Data availability delays (days) - realistic delays for financial data
        'data_availability_delays': {
            '재무정보': 45,     # Financial reports: 45 days after quarter/year end
            '수출실적': 30,         # Trade statistics: 30 days comprehensive data
            '신용등급': 7,          # Credit ratings: 7 days processing delay
            'GDP': 45,           # GDP data: 45 days after quarter end
            '총수출': 30,         # Trade statistics: 30 days comprehensive data
            '업종': 0,       # Industry codes: static/immediate (no delay)
            '환변동': 1,       # Exchange rate data: 1 day after daily data
        },
        
        # Apply availability delays to make model production-realistic
        'apply_availability_delays': False,
        
        # Additional safety margin for data processing/transmission delays
        'safety_margin_days': 7,  # Extra 7 days buffer for real-world conditions
    }


# Main execution example
if __name__ == "__main__":
    
    # Example usage
    config = get_example_config()
    
    # Create dataset
    creator = DatasetCreator(config)
    dataset = creator.create_dataset()
    
    # Save result
    creator.save_dataset('../data/processed/credit_risk_dataset.csv')
    
    logger.info("Dataset creation completed!")
    logger.info("Ready for XGBoost training!")