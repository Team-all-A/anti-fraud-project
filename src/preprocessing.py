import polars as pl

def load_and_merge(transaction_path: str, users_path: str) -> pl.DataFrame:
    """Downloading and merging transactions with users"""
    transactions = pl.read_csv(transaction_path)
    users = pl.read_csv(users_path)
    df = transactions.join(users, on='id_user', how='left')
    return df

def handle_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Filling up missing data: numbers - median, lines - 'unknown'."""
    numeric_cols = [
        col for col , dtype in zip(df.columns, df.dtypes)
        if dtype in (pl.Float64, pl.Int64, pl.Float32, pl.Int32)
    ]
    string_cols = [
        col for col, dtype in zip(df.columns, df.dtypes)
        if dtype == pl.Utf8
    ]

    df = df.with_columns([
        pl.col(col).fill_null(df[col].median())
        for col in numeric_cols
    ])
    df = df.with_columns([
        pl.col(col).fill_null('unknown')
        for col in string_cols
    ])
    return df

def prepare_features(df: pl.DataFrame, target_col: str = None):
    """
    Повертає X (numpy) і y (numpy або None для тестових даних).
    Міша додаватиме свої фічі перед цим кроком.
    """

    drop_cols = ['id_user']
    if target_col and target_col in df.columns:
        drop_cols.append(target_col)

    X = df.drop(drop_cols).to_numpy()
    y = df[target_col].to_numpy() if target_col in df.columns else None
    return X, y