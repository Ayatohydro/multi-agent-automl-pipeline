
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a tabular dataset from a CSV file.
    This acts as a simple 'custom tool' used by agents.
    """
    df = pd.read_csv(csv_path)
    return df


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Very simple heuristic to decide if task is classification or regression.
    """
    n_unique = df[target_col].nunique()

    # If many unique values, assume regression
    if pd.api.types.is_numeric_dtype(df[target_col]) and n_unique > 20:
        return "regression"
    else:
        return "classification"


def basic_train_val_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into train/validation sets.
    Uses stratify only when it's safe to do so.
    Returns: X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Encoding Categorical columns into Numbers
    X = pd.get_dummies(X, drop_first=True)

    n_samples = len(df)
    n_classes = y.nunique()

    # Decide whether stratify is safe
    stratify_target = None
    if n_classes > 1:
        test_count = int(n_samples * test_size)
        if test_count >= n_classes:
            stratify_target = y  # safe to stratify

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
    return X_train, X_val, y_train, y_val
