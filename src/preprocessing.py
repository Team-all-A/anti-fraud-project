import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

# Constants
ID_COLS = ['id_user', 'card_mask_hash', 'card_holder', 'email']
DATE_COLS = ['timestamp_tr', 'timestamp_reg']

# Downloading and merging transactions with users
def load_and_merge(transaction_path: str, users_path: str) -> pl.DataFrame:
    transactions = pl.read_csv(transaction_path, infer_schema_length=10000)
    users = pl.read_csv(users_path, infer_schema_length=10000)
    df = transactions.join(users, on='id_user', how='left')
    return df

# Handling missing values with isolated state for cross-validation
class PolarsImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_medians = {}
        self.string_fills = 'unknown'

    def fit(self, X: pl.DataFrame, y=None):
        numeric_cols = [
            col for col, dtype in zip(X.columns, X.dtypes)
            if dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
        ]
        
        for col in numeric_cols:
            median_val = X[col].median()
            self.numeric_medians[col] = median_val if median_val is not None else 0
            
        return self
    
    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        
        string_cols = [
            col for col, dtype in zip(df.columns, df.dtypes) 
            if dtype == pl.Utf8
        ]
        
        df = df.with_columns([
            pl.col(col).fill_null(self.string_fills) for col in string_cols
        ])
        
        for col, median_val in self.numeric_medians.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).fill_null(median_val))
                
        return df

# Parsing datetime features without state retention
class PolarsDatetimeParser(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols: list):
        self.date_cols = date_cols

    def fit(self, X: pl.DataFrame, y=None):
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        df = X.clone()
        
        for col in self.date_cols:
            if col in df.columns:
                # Parsing string to datetime and removing timezone
                df = df.with_columns(
                    pl.col(col)
                    .str.to_datetime(strict=False)
                    .dt.replace_time_zone(None)
                    .alias(col)
                )
                
                # Extracting datetime components into separate columns
                df = df.with_columns([
                    pl.col(col).dt.year().alias(f'{col}_year'),
                    pl.col(col).dt.month().alias(f'{col}_month'),
                    pl.col(col).dt.day().alias(f'{col}_day'),
                    pl.col(col).dt.hour().alias(f'{col}_hour'),
                    pl.col(col).dt.weekday().alias(f'{col}_weekday'),
                ]).drop(col)
                
        return df