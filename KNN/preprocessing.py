import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)


def clean_data(df, columns_to_drop):
    """Remove unnecessary columns."""
    return df.drop(columns=columns_to_drop)


def split_features_target(df, target_column):
    """Separate features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def encode_target(y):
    """Convert categorical target values into numbers."""
    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    return y_encoded, encoder


def split_data(X, y, test_size, random_state):
    """Split data into training and testing sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def scale_features(X_train, X_test):
    """Standardize numerical features."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler