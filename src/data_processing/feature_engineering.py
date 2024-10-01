import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


def engineer_features(df):
    # Convert 'Timestamp' to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Time-based features
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    # Statistical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10).std()
        df[f'{col}_kurtosis'] = df[col].rolling(window=50).apply(kurtosis)
        df[f'{col}_skew'] = df[col].rolling(window=50).apply(skew)

    # Interaction features
    df['BytesPerPacket'] = df['Total Length of Fwd Packets'] / \
        df['Total Fwd Packets']
    df['PacketsPerSecond'] = df['Total Fwd Packets'] / df['Flow Duration']

    # Extract time series features using tsfresh
    ts_features = extract_features(df, column_id="Flow ID", column_sort="Timestamp",
                                   default_fc_parameters=MinimalFCParameters())

    return pd.concat([df, ts_features], axis=1)
